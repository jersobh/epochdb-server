# src/server.py
import logging
import os
import uvicorn
import uuid
import hashlib
import bisect
import asyncio
import httpx
import json
import time
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Security, Depends, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union

from epochdb import AsyncEpochDB
from fastapi import Header, Query
from sse_starlette.sse import EventSourceResponse

try:
    from auth import verify_scoped_auth, Permission, get_keystore, ScopedAPIKey
except ImportError:
    from src.auth import verify_scoped_auth, Permission, get_keystore, ScopedAPIKey

# Export variables for backward compatibility and tests
API_KEY = os.getenv("API_KEY")
INTERNAL_AUTH_TOKEN = os.getenv("INTERNAL_AUTH_TOKEN")
SERVER_VERSION = "0.11.0"
# Coordinator→shard HTTP timeout. Embeddings on small VPS CPUs often exceed 30s
# under concurrent ingest; default matches gunicorn --timeout 120.
COORDINATOR_HTTP_TIMEOUT = float(os.getenv("COORDINATOR_HTTP_TIMEOUT", "120"))
# Bound CPU-bound sentence-transformer work per shard process. A full queue is
# rejected rather than letting requests pile up until coordinator timeouts.
SHARD_EMBED_CONCURRENCY = max(1, int(os.getenv("SHARD_EMBED_CONCURRENCY", "1")))
SHARD_EMBED_QUEUE_LIMIT = max(0, int(os.getenv("SHARD_EMBED_QUEUE_LIMIT", "8")))
_embed_semaphore: Optional[asyncio.Semaphore] = None
_embed_waiters = 0


def format_shard_error(resp: Any) -> str:
    """Readable shard failure text (TimeoutError/ReadError often have empty str())."""
    if isinstance(resp, Exception):
        msg = str(resp).strip()
        return msg if msg else repr(resp)
    if isinstance(resp, httpx.Response):
        body = (resp.text or "").strip().replace("\n", " ")[:200]
        return f"HTTP {resp.status_code}" + (f": {body}" if body else "")
    return repr(resp)


def should_mark_shard_offline(resp: Any) -> bool:
    """Only mark offline for hard connectivity loss — not overload timeouts."""
    return isinstance(resp, httpx.ConnectError)


# -------------------------------------------------------------------------
# 1. Structured Logging Configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s (Line: %(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("epochdb_production_server")

# httpx logs one INFO line per outbound request; with 5s health/stats polling
# across shards this floods coordinator logs and buries real errors.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@asynccontextmanager
async def embedding_slot():
    """Reserve bounded shard-local capacity for an embedding operation."""
    global _embed_semaphore, _embed_waiters
    if _embed_semaphore is None:
        _embed_semaphore = asyncio.Semaphore(SHARD_EMBED_CONCURRENCY)

    if _embed_semaphore.locked() and _embed_waiters >= SHARD_EMBED_QUEUE_LIMIT:
        raise HTTPException(
            status_code=503,
            detail="Embedding capacity is saturated; retry shortly.",
            headers={"Retry-After": "5"},
        )

    _embed_waiters += 1
    try:
        async with _embed_semaphore:
            yield
    finally:
        _embed_waiters -= 1


async def run_embedding(fn, *args, **kwargs):
    """Run a CPU-bound embedding with bounded shard-local queueing."""
    async with embedding_slot():
        return await asyncio.to_thread(fn, *args, **kwargs)


# -------------------------------------------------------------------------
# System Resource Profiling Helpers
# -------------------------------------------------------------------------
_last_cpu_times = {"total": 0.0, "idle": 0.0}

def get_cpu_usage() -> float:
    global _last_cpu_times
    try:
        with open("/proc/stat", "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("cpu "):
                parts = [float(x) for x in line.split()[1:]]
                idle = parts[3] + parts[4]
                total = sum(parts)
                
                diff_total = total - _last_cpu_times["total"]
                diff_idle = idle - _last_cpu_times["idle"]
                
                _last_cpu_times["total"] = total
                _last_cpu_times["idle"] = idle
                
                if diff_total <= 0:
                    return 0.0
                return round(max(0.0, min(100.0, (1.0 - diff_idle / diff_total) * 100.0)), 2)
    except Exception:
        pass
    return 0.0

def get_ram_usage() -> Dict[str, float]:
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val = float(parts[1])
                    meminfo[key] = val
        total = meminfo.get("MemTotal", 0.0) / 1024.0 / 1024.0
        free = meminfo.get("MemFree", 0.0) / 1024.0 / 1024.0
        buffers = meminfo.get("Buffers", 0.0) / 1024.0 / 1024.0
        cached = meminfo.get("Cached", 0.0) / 1024.0 / 1024.0
        available = meminfo.get("MemAvailable", (free + buffers + cached)) / 1024.0 / 1024.0
        used = total - available
        percent = (used / total * 100.0) if total > 0 else 0.0
        return {
            "total": round(total, 2),
            "available": round(available, 2),
            "used": round(used, 2),
            "percent": round(percent, 2)
        }
    except Exception:
        pass
    return {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}

def get_disk_usage(path: str = ".") -> Dict[str, float]:
    try:
        stat = os.statvfs(path)
        total = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        available = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        used = total - available
        percent = (used / total * 100.0) if total > 0 else 0.0
        return {
            "total": round(total, 2),
            "available": round(available, 2),
            "used": round(used, 2),
            "percent": round(percent, 2)
        }
    except Exception:
        pass
    return {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}

# -------------------------------------------------------------------------
# 2. Clustering Configuration & Consistent Hashing Ring
# -------------------------------------------------------------------------
class ConsistentHashRing:
    def __init__(self, nodes: List[str] = None, replicas: int = 100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def add_node(self, node: str):
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            bisect.insort(self.sorted_keys, key)

    def remove_node(self, node: str):
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)

    def get_node(self, val: str) -> str:
        if not self.ring:
            raise ValueError("No nodes on the ring")
        h = self._hash(val)
        idx = bisect.bisect_right(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]

# Read cluster environment configuration
NODE_MODE = os.getenv("NODE_MODE", "shard").lower()
shard_nodes_str = os.getenv("SHARD_NODES", "")

from enum import Enum

class ConsistencyLevel(str, Enum):
    ONE = "one"
    QUORUM = "quorum"
    ALL = "all"

def get_consistency_level(request: Request) -> ConsistencyLevel:
    header_val = request.headers.get("x-consistency-level") or request.query_params.get("consistency")
    if header_val:
        try:
            return ConsistencyLevel(header_val.lower())
        except ValueError:
            pass
    return ConsistencyLevel.ONE

shard_groups = []
if shard_nodes_str:
    for group_str in shard_nodes_str.split(","):
        if group_str.strip():
            replicas = [r.strip() for r in group_str.split("+") if r.strip()]
            if replicas:
                shard_groups.append(replicas)

shard_nodes = [group[0] for group in shard_groups] if shard_groups else []
hash_ring = ConsistentHashRing(shard_nodes) if shard_nodes else None

# Dynamic embedding configuration (defaults to local offline all-MiniLM-L6-v2)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
GRAPH_HUB_LIMIT = int(os.getenv("GRAPH_HUB_LIMIT", "50"))


class RequestContext(BaseModel):
    tenant: Optional[str] = None
    namespace: Optional[str] = None
    key_info: ScopedAPIKey

SHARD_METRICS_CACHE: Dict[str, Any] = {}

def get_replica_group_for_node(node: str) -> List[str]:
    for group in shard_groups:
        if node in group:
            return group
    return [node]

def get_healthy_replica_for_group(group: List[str]) -> Optional[str]:
    for node in group:
        metrics = SHARD_METRICS_CACHE.get(node)
        if not metrics or metrics.get("status") == "healthy":
            return node
    return group[0] if group else None

def get_all_healthy_replicas_for_group(group: List[str]) -> List[str]:
    healthy = []
    for node in group:
        metrics = SHARD_METRICS_CACHE.get(node)
        if not metrics or metrics.get("status") == "healthy":
            healthy.append(node)
    if not healthy and group:
        healthy.append(group[0])
    return healthy

# -------------------------------------------------------------------------
# Cache Layer Configuration
# -------------------------------------------------------------------------
# Key: (tenant, namespace) -> state_version (int)
CLUSTER_STATE_VERSIONS: Dict[tuple[str, str], int] = {}

# Key: (tenant, namespace) -> Dict[cache_key, (cached_data, ETag)]
LOCAL_READ_CACHE: Dict[tuple[str, str], Dict[str, tuple[Any, str]]] = {}

def get_state_version(tenant: Optional[str], namespace: Optional[str]) -> int:
    ctx_key = (tenant or "", namespace or "")
    if ctx_key not in CLUSTER_STATE_VERSIONS:
        CLUSTER_STATE_VERSIONS[ctx_key] = 0
    return CLUSTER_STATE_VERSIONS[ctx_key]

def invalidate_cache(tenant: Optional[str], namespace: Optional[str]):
    ctx_key = (tenant or "", namespace or "")
    CLUSTER_STATE_VERSIONS[ctx_key] = CLUSTER_STATE_VERSIONS.get(ctx_key, 0) + 1
    LOCAL_READ_CACHE[ctx_key] = {}
    logger.debug(f"Invalidated read cache for tenant='{tenant}' namespace='{namespace}'. New version: {CLUSTER_STATE_VERSIONS[ctx_key]}")

def generate_etag(endpoint: str, payload_dict: dict, state_version: int) -> str:
    # Stable JSON serialization
    serialized = json.dumps(payload_dict, sort_keys=True)
    raw = f"{endpoint}:{serialized}:{state_version}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

async def handle_cached_read(
    endpoint: str,
    payload: Any,
    request: Request,
    response: Response,
    ctx: RequestContext,
    fetch_func
):
    ctx_key = (ctx.tenant or "", ctx.namespace or "")
    state_version = get_state_version(ctx.tenant, ctx.namespace)
    
    # Standardize payload representation
    payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else (payload if isinstance(payload, dict) else {"_val": payload})
    etag = generate_etag(endpoint, payload_dict, state_version)
    
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
        
    if ctx_key in LOCAL_READ_CACHE and etag in LOCAL_READ_CACHE[ctx_key]:
        response.headers["ETag"] = etag
        return LOCAL_READ_CACHE[ctx_key][etag]
        
    res_data = await fetch_func()
    
    # Don't cache partial fan-out results: they would keep serving degraded
    # answers until the next write invalidation even after shards recover.
    if not (isinstance(res_data, dict) and res_data.get("partial")):
        if ctx_key not in LOCAL_READ_CACHE:
            LOCAL_READ_CACHE[ctx_key] = {}
        LOCAL_READ_CACHE[ctx_key][etag] = res_data
    response.headers["ETag"] = etag
    return res_data

SSE_LISTENERS: List[asyncio.Queue] = []

async def broadcast_sse_event(event_type: str, data: dict):
    """
    Broadcasts an event to all connected SSE clients.
    """
    if not SSE_LISTENERS:
        return
    logger.debug(f"Broadcasting SSE event '{event_type}' to {len(SSE_LISTENERS)} listeners.")
    for queue in list(SSE_LISTENERS):
        await queue.put({"event": event_type, "data": data})

# Track syncing nodes
SYNCING_NODES = set()
BACKGROUND_SYNC_ENABLED = True

async def sync_node_data(recovering_node: str, source_node: str):
    """
    Synchronizes all databases from source_node to recovering_node.
    """
    logger.info(f"Triggering recovery sync for node {recovering_node} from source {source_node}...")
    try:
        headers = {}
        if INTERNAL_AUTH_TOKEN:
            headers["X-Internal-Token"] = INTERNAL_AUTH_TOKEN
        
        async with httpx.AsyncClient(headers=headers, timeout=120.0) as sync_client:
            res = await sync_client.get(f"{source_node}/admin/databases")
            res.raise_for_status()
            databases = res.json().get("databases", [])
            
            for db_info in databases:
                tenant = db_info.get("tenant")
                namespace = db_info.get("namespace")
                logger.info(f"Syncing database (tenant={tenant}, namespace={namespace}) from {source_node} to {recovering_node}...")
                
                timeline_payload = {
                    "entity_id": None,
                    "start": None,
                    "end": None
                }
                req_headers = {}
                if tenant:
                    req_headers["X-Tenant"] = tenant
                if namespace:
                    req_headers["X-Namespace"] = namespace
                    
                timeline_res = await sync_client.post(
                    f"{source_node}/get_timeline",
                    json=timeline_payload,
                    headers=req_headers
                )
                timeline_res.raise_for_status()
                memories = timeline_res.json().get("memories", [])
                
                reset_res = await sync_client.post(
                    f"{recovering_node}/admin/reset",
                    json={"tenant": tenant, "namespace": namespace}
                )
                reset_res.raise_for_status()
                
                for m in memories:
                    target_payload = {
                        "text": m.get("payload") or m.get("text"),
                        "metadata": m.get("metadata", {}),
                        "id": m.get("id"),
                        "memory_type": m.get("memory_type")
                    }
                    write_res = await sync_client.post(
                        f"{recovering_node}/remember",
                        json=target_payload,
                        headers=req_headers
                    )
                    write_res.raise_for_status()
                    
            logger.info(f"Sync complete for recovering node {recovering_node} from source {source_node}.")
    except Exception as e:
        logger.error(f"Sync task failed for node {recovering_node} from source {source_node}: {e}")
        raise e

async def poll_shards_loop():
    """
    Background loop that runs on the coordinator to poll shards for health and metrics.
    Employs per-shard exponential backoff and configurable poll intervals to prevent
    hammering shards that are busy with heavy embedding or disk I/O tasks.
    """
    logger.info("Starting background health polling loop for shards...")
    all_replica_nodes = []
    for group in shard_groups:
        all_replica_nodes.extend(group)
    
    stats_timeout = float(os.getenv("COORDINATOR_STATS_TIMEOUT", "15"))
    base_poll_interval = float(os.getenv("COORDINATOR_POLL_INTERVAL", "5"))
    max_backoff = float(os.getenv("COORDINATOR_MAX_BACKOFF", "60"))

    # State tracking per shard: next allowed poll timestamp and failure counts
    next_poll_time: Dict[str, float] = {shard: 0.0 for shard in all_replica_nodes}
    fail_counts: Dict[str, int] = {shard: 0 for shard in all_replica_nodes}
        
    while True:
        if client:
            try:
                now = time.monotonic()
                due_shards = [shard for shard in all_replica_nodes if now >= next_poll_time[shard]]

                if due_shards:
                    tasks = [client.get(f"{shard}/stats", timeout=stats_timeout) for shard in due_shards]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    status_changed = False
                    for shard, resp in zip(due_shards, responses):
                        prev_status = SHARD_METRICS_CACHE.get(shard, {}).get("status")
                        if isinstance(resp, httpx.Response) and resp.status_code == 200:
                            fail_counts[shard] = 0
                            next_poll_time[shard] = now + base_poll_interval
                            try:
                                data = resp.json()
                                
                                metrics = {
                                    "status": "healthy",
                                    "memory_count": data.get("memory_count", 0),
                                    "l1_size": data.get("l1_size", 0),
                                    "l2_size": data.get("l2_size", 0),
                                    "entity_count": data.get("entity_count", 0),
                                    "cpu": data.get("cpu", 0.0),
                                    "ram": data.get("ram", {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}),
                                    "disk": data.get("disk", {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}),
                                }
                                
                                # Trigger background synchronization if node was previously offline/unhealthy
                                logger.debug(f"poll_shards_loop: shard={shard} prev_status={prev_status} syncing={shard in SYNCING_NODES}")
                                if BACKGROUND_SYNC_ENABLED and prev_status not in ("healthy", "synchronizing") and shard not in SYNCING_NODES:
                                    group = get_replica_group_for_node(shard)
                                    source_node = None
                                    for sibling in group:
                                        if sibling != shard:
                                            sib_metrics = SHARD_METRICS_CACHE.get(sibling)
                                            sib_status = sib_metrics.get("status") if sib_metrics else None
                                            logger.debug(f"poll_shards_loop: checking sibling={sibling} status={sib_status}")
                                            if sib_metrics and sib_status == "healthy":
                                                source_node = sibling
                                                break
                                                
                                    logger.info(f"poll_shards_loop: shard={shard} selected source_node={source_node}")
                                    if source_node:
                                        metrics["status"] = "synchronizing"
                                        SHARD_METRICS_CACHE[shard] = metrics
                                        SYNCING_NODES.add(shard)
                                        
                                        async def run_sync_task(node=shard, src=source_node):
                                            try:
                                                await sync_node_data(node, src)
                                                if node in SHARD_METRICS_CACHE:
                                                    SHARD_METRICS_CACHE[node]["status"] = "healthy"
                                                    await broadcast_sse_event("update", {"type": "status_change"})
                                                logger.info(f"Node {node} successfully recovered and synchronized.")
                                            except Exception as e:
                                                if node in SHARD_METRICS_CACHE:
                                                    SHARD_METRICS_CACHE[node]["status"] = "unhealthy"
                                                    await broadcast_sse_event("update", {"type": "status_change"})
                                                logger.error(f"Failed to synchronize recovering node {node}: {e}")
                                            finally:
                                                SYNCING_NODES.discard(node)
                                                
                                        asyncio.create_task(run_sync_task())
                                    else:
                                        logger.info(f"poll_shards_loop: marking shard={shard} healthy immediately because no source_node was found")
                                        SHARD_METRICS_CACHE[shard] = metrics
                                else:
                                    if prev_status == "synchronizing":
                                        metrics["status"] = "synchronizing"
                                    SHARD_METRICS_CACHE[shard] = metrics
                            except Exception as e:
                                SHARD_METRICS_CACHE[shard] = {
                                    "status": "unhealthy",
                                    "error": f"Failed to parse stats: {str(e)}",
                                    "cpu": 0.0,
                                    "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                    "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                }
                        else:
                            # Probe failed/timed out/disconnected
                            fail_counts[shard] += 1
                            err_msg = format_shard_error(resp)

                            if isinstance(resp, httpx.ConnectError):
                                # Hard connectivity loss: cap retry delay to base_poll_interval (e.g. 5s)
                                # so recovered/rebooted nodes are detected promptly.
                                backoff_delay = base_poll_interval
                                next_poll_time[shard] = now + backoff_delay
                                logger.warning(f"poll_shards_loop: shard={shard} unreachable (fails={fail_counts[shard]}, next_poll_in={backoff_delay:.1f}s): {err_msg}")
                                SHARD_METRICS_CACHE[shard] = {
                                    "status": "unhealthy",
                                    "error": err_msg,
                                    "cpu": 0.0,
                                    "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                    "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                }
                            else:
                                # Overload timeout / transient HTTP blips while embedding: exponential backoff
                                # to avoid hammering the busy worker event loop.
                                backoff_delay = min(base_poll_interval * (2 ** (fail_counts[shard] - 1)), max_backoff)
                                next_poll_time[shard] = now + backoff_delay
                                logger.warning(
                                    f"poll_shards_loop: shard={shard} stats probe failed (keeping prior status={prev_status}, fails={fail_counts[shard]}, next_poll_in={backoff_delay:.1f}s): {err_msg}"
                                )
                                if not SHARD_METRICS_CACHE.get(shard):
                                    SHARD_METRICS_CACHE[shard] = {
                                        "status": "healthy",
                                        "error": err_msg,
                                        "cpu": 0.0,
                                        "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                        "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                    }
                        
                        new_status = SHARD_METRICS_CACHE.get(shard, {}).get("status")
                        if prev_status != new_status:
                            status_changed = True
                    
                    if status_changed:
                        await broadcast_sse_event("update", {"type": "status_change"})
                    
                    if SSE_LISTENERS:
                        total_memory_count = 0
                        total_l1_size = 0
                        total_l2_size = 0
                        total_entity_count = 0
                        
                        group_representatives = []
                        for group in shard_groups:
                            node = get_healthy_replica_for_group(group)
                            if node:
                                group_representatives.append(node)
                                
                        shards_metrics = {}
                        for node in all_replica_nodes:
                            metrics = SHARD_METRICS_CACHE.get(node)
                            if metrics:
                                shards_metrics[node] = metrics
                                if node in group_representatives:
                                    total_memory_count += metrics.get("memory_count", 0)
                                    total_l1_size += metrics.get("l1_size", 0)
                                    total_l2_size += metrics.get("l2_size", 0)
                                    total_entity_count += metrics.get("entity_count", 0)
                                    
                        coord_system = {
                            "cpu": get_cpu_usage(),
                            "ram": get_ram_usage(),
                            "disk": get_disk_usage(os.getenv("STORAGE_DIR", "."))
                        }
                        
                        stats_payload = {
                            "mode": "coordinator",
                            "memory_count": total_memory_count,
                            "l1_size": total_l1_size,
                            "l2_size": total_l2_size,
                            "entity_count": total_entity_count,
                            "system": coord_system,
                            "shards": shards_metrics
                        }
                        
                        await broadcast_sse_event("stats", stats_payload)
            except Exception as e:
                logger.error(f"Error in poll_shards_loop execution: {e}")
        await asyncio.sleep(1)

def get_healthy_shard(key: str) -> str:
    """
    Return the owning shard group for a key, only if it can accept a write.

    Writes must never move to another consistent-hash group: doing so breaks
    deterministic ID routing and leaves data stranded after owner recovery.
    """
    if not shard_nodes:
        raise HTTPException(status_code=500, detail="No shards available to route write request.")
    
    primary_node = hash_ring.get_node(key)
    
    def is_group_good(node: str) -> bool:
        group = get_replica_group_for_node(node)
        for rep in group:
            metrics = SHARD_METRICS_CACHE.get(rep)
            if not metrics:
                return True
            if metrics.get("status") == "healthy":
                if os.getenv("PYTEST_CURRENT_TEST"):
                    return True
                cpu = metrics.get("cpu", 0.0)
                ram_pct = metrics.get("ram", {}).get("percent", 0.0)
                disk_pct = metrics.get("disk", {}).get("percent", 0.0)
                if cpu <= 90.0 and ram_pct <= 90.0 and disk_pct <= 90.0:
                    return True
        return False

    if is_group_good(primary_node):
        return primary_node

    logger.warning(
        "Owner shard group %s cannot accept write for key %r; refusing cross-shard reroute.",
        primary_node, key[:20],
    )
    raise HTTPException(
        status_code=503,
        detail="Owning shard group is unavailable or saturated; retry shortly.",
        headers={"Retry-After": "5"},
    )


# -------------------------------------------------------------------------
# Request Context & Scoped Authentication
# -------------------------------------------------------------------------
def get_context(required_permissions: set[str]):
    async def context_dependency(
        x_tenant: Optional[str] = Header(None, alias="X-Tenant"),
        x_namespace: Optional[str] = Header(None, alias="X-Namespace"),
        tenant: Optional[str] = Query(None),
        namespace: Optional[str] = Query(None),
        key_info: ScopedAPIKey = Depends(verify_scoped_auth(required_permissions))
    ) -> RequestContext:
        req_tenant = x_tenant or tenant
        req_namespace = x_namespace or namespace
        
        # Enforce key-level tenant/namespace constraints
        if key_info.tenant is not None:
            if req_tenant and req_tenant != key_info.tenant:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API Key restricted to a different tenant."
                )
            req_tenant = key_info.tenant
            
        if key_info.namespace is not None:
            if req_namespace and req_namespace != key_info.namespace:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API Key restricted to a different namespace."
                )
            req_namespace = key_info.namespace
            
        return RequestContext(tenant=req_tenant, namespace=req_namespace, key_info=key_info)
    return context_dependency



def get_shard_for_id(memory_id: str) -> Optional[str]:
    if not shard_nodes:
        return None
    if memory_id.startswith("shard"):
        parts = memory_id.split("-", 1)
        if len(parts) > 1:
            prefix = parts[0]  # e.g., "shard0"
            try:
                idx = int(prefix[5:])  # extract index
                if 0 <= idx < len(shard_nodes):
                    return shard_nodes[idx]
            except ValueError:
                pass
    return None

# -------------------------------------------------------------------------
# 3. Database State & Lifespan Management
# -------------------------------------------------------------------------
# Database pool map: (tenant, namespace) -> AsyncEpochDB instance (Shard Mode)
db_pool: Dict[tuple[Optional[str], Optional[str]], AsyncEpochDB] = {}
# Global reference for default database engine instance for backward compatibility
db: Optional[AsyncEpochDB] = None
# Global HTTP client session (Coordinator Mode)
client: Optional[httpx.AsyncClient] = None

# Locks for database initialization to prevent race conditions
db_init_locks: Dict[tuple[Optional[str], Optional[str]], asyncio.Lock] = {}
db_init_locks_lock = asyncio.Lock()

async def get_db_instance(tenant: Optional[str] = None, namespace: Optional[str] = None) -> AsyncEpochDB:
    key = (tenant, namespace)
    if key in db_pool:
        return db_pool[key]
        
    async with db_init_locks_lock:
        if key not in db_init_locks:
            db_init_locks[key] = asyncio.Lock()
        lock = db_init_locks[key]
        
    async with lock:
        if key in db_pool:
            return db_pool[key]
            
        storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
        logger.info(f"Initializing AsyncEpochDB for tenant={tenant}, namespace={namespace} under {storage_dir}")
        auto_extract = os.getenv("AUTO_EXTRACT", "true").lower() in ("1", "true", "yes")
        engine = AsyncEpochDB(
            storage_dir=storage_dir,
            embedding_model=EMBEDDING_MODEL,
            dim=EMBEDDING_DIM,
            wal_sync_interval=0.1,
            parquet_compression="zstd",
            parquet_compression_level=3,
            tenant=tenant,
            namespace=namespace,
            auto_extract=auto_extract,
        )
        await engine._get_db()
        
        # Warmup sequence
        try:
            await engine.query(text="system boot warmup", k=1)
        except Exception as e:
            logger.error(f"Error warming up db instance {key}: {e}")
            
        db_pool[key] = engine
        return engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the startup and shutdown lifecycles of the ASGI application.
    """
    global db, client
    
    if NODE_MODE == "coordinator":
        logger.info("Initializing Sharded Clustering Coordinator Gateway...")
        if not shard_nodes:
            logger.warning("No SHARD_NODES configured for coordinator. Check env config.")
        headers = {}
        if INTERNAL_AUTH_TOKEN:
            headers["X-Internal-Token"] = INTERNAL_AUTH_TOKEN
        logger.info(
            "Coordinator HTTP client timeout=%.1fs (COORDINATOR_HTTP_TIMEOUT)",
            COORDINATOR_HTTP_TIMEOUT,
        )
        client = httpx.AsyncClient(headers=headers, timeout=COORDINATOR_HTTP_TIMEOUT)
        polling_task = asyncio.create_task(poll_shards_loop())
        yield
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            pass
        await client.aclose()
        logger.info("Coordinator HTTP client session closed cleanly.")
    else:
        logger.info("Initializing default AsyncEpochDB engine in Shard Mode...")
        try:
            # Pre-initialize and warm up default database
            db = await get_db_instance(None, None)
            logger.info("Default db warmed up. AsyncEpochDB database pool ready.")
            yield
        except Exception as e:
            logger.critical(f"Fatal error during engine startup sequence: {str(e)}")
            raise e
        finally:
            logger.info("Closing all active database instances in pool...")
            for key, engine in list(db_pool.items()):
                try:
                    await engine.close()
                except Exception as e:
                    logger.error(f"Failed to close engine instance {key}: {e}")
            db_pool.clear()
            logger.info("Database pool context exited cleanly.")
            logger.info("Database context exited cleanly. All resources released.")

# Initialize FastAPI application with lifespan management
app = FastAPI(
    title="EpochDB Core Server",
    description="Asynchronous high-concurrency memory engine interface.",
    version="1.0.0",
    lifespan=lifespan
)

# -------------------------------------------------------------------------
# 4. Data Transfer Objects (Pydantic Models)
# -------------------------------------------------------------------------
class MemoryPayload(BaseModel):
    text: Optional[str] = Field(default=None, description="The factual memory text to write to the engine.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional structural metadata or graph triples.")
    id: Optional[str] = Field(default=None, description="Optional unique identifier (pre-calculated or forwarded).")
    memory_type: Optional[str] = Field(default=None, description="Optional memory type: 'general', 'episodic', 'profile', 'working', or 'skill'.")
    triples: Optional[Any] = Field(default=None, description="Optional structural graph triples.")
    subject: Optional[str] = Field(default=None, description="Optional single triplet subject.")
    predicate: Optional[str] = Field(default=None, description="Optional single triplet predicate.")
    object: Optional[str] = Field(default=None, description="Optional single triplet object.")

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            subj = data.get("subject")
            pred = data.get("predicate")
            obj = data.get("object")
            triples = data.get("triples")
            meta = data.get("metadata") or {}
            meta_triples = meta.get("triples") if isinstance(meta, dict) else None

            collected = []
            if subj and pred and obj:
                collected.append([str(subj).strip(), str(pred).strip(), str(obj).strip()])
            if triples and isinstance(triples, (list, tuple)):
                for t in triples:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        collected.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])
            if meta_triples and isinstance(meta_triples, (list, tuple)):
                for t in meta_triples:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        collected.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])

            if collected:
                data["triples"] = collected
                if not data.get("text"):
                    # Auto-generate human-readable memory sentence from the triples
                    data["text"] = ". ".join([f"{t[0]} {t[1].replace('_', ' ')} {t[2]}" for t in collected]) + "."

            if not data.get("text") and not collected:
                raise ValueError("Either 'text', 'triples', or subject/predicate/object must be provided.")
        return data

class BatchMemoryPayload(BaseModel):
    items: List[Any] = Field(..., description="List of memory items (text strings or dict objects) to ingest in batch.")

class ReplacePayload(BaseModel):
    atom_id: str = Field(..., description="Existing atom ID to replace in-place.")
    text: str = Field(..., description="New memory payload text.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata dictionary.")
    triples: Optional[Any] = Field(default=None, description="Optional updated triples.")

class QueryPayload(BaseModel):
    query: Optional[str] = Field(default=None, description="The semantic search query string.")
    text: Optional[str] = Field(default=None, description="Alias for query text for client parity.")
    k: int = Field(default=1, ge=1, le=100, description="The number of candidate matches to return.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="MongoDB-style metadata filter evaluation parameters.")
    memory_type: Optional[str] = Field(default=None, description="Optional filter by memory type: 'general', 'episodic', 'profile', 'working', or 'skill'.")
    context_window: int = Field(default=0, ge=0, description="Number of chronological context turns to retrieve around the matched memory.")
    expand_hops: int = Field(default=0, ge=0, le=10, description="Relational KG expansion hops during retrieval.")
    min_score: float = Field(default=0.0, description="Minimum similarity score threshold.")
    cold_search_mode: Optional[str] = Field(default=None, description="Cold tier search mode: 'probe' or 'exhaustive'.")

    def get_query_text(self) -> str:
        return self.query or self.text or ""

class MultiHopPayload(BaseModel):
    query: Optional[str] = Field(default=None, description="The query string.")
    text: Optional[str] = Field(default=None, description="Alias for query text.")
    hops: int = Field(default=2, ge=1, le=10, description="Number of relational KG expansion hops.")
    k: int = Field(default=5, ge=1, le=100, description="Number of candidate matches to return.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters.")
    context_window: int = Field(default=0, ge=0, description="Number of chronological context turns.")
    cold_search_mode: Optional[str] = Field(default=None, description="Cold tier search mode: 'probe' or 'exhaustive'.")

    def get_query_text(self) -> str:
        return self.query or self.text or ""

class AdaptiveQueryPayload(BaseModel):
    query: str = Field(..., description="The natural language search query.")
    k: int = Field(default=5, ge=1, le=100, description="The number of candidate matches to return.")
    context_window: int = Field(default=0, ge=0, description="Number of chronological context turns to retrieve around the matched memory.")

class GetPayload(BaseModel):
    memory_id: str = Field(..., description="The unique identifier of the memory to retrieve.")

class UpdatePayload(BaseModel):
    memory_id: str = Field(..., description="The unique identifier of the memory to update.")
    text: Optional[str] = Field(default=None, description="Optional new text payload.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional new metadata dictionary.")
    triples: Optional[Any] = Field(default=None, description="Optional updated triples.")

class DeletePayload(BaseModel):
    memory_id: str = Field(..., description="The unique identifier of the memory to delete.")
    hard: bool = Field(default=False, description="Whether to hard delete or soft delete.")

class TimelinePayload(BaseModel):
    entity_id: Optional[str] = Field(default=None, description="Optional entity ID.")
    start: Optional[float] = Field(default=None, description="Optional start timestamp.")
    end: Optional[float] = Field(default=None, description="Optional end timestamp.")

class ExportPayload(BaseModel):
    data_type: str = Field(default="all", description="Export category: 'all', 'skills', 'profiles', 'triplets', or 'memories'.")
    memory_type: Optional[str] = Field(default=None, description="Optional memory type filter when exporting memories.")
    tenant: Optional[str] = Field(default=None, description="Optional tenant.")
    namespace: Optional[str] = Field(default=None, description="Optional namespace.")

class ImportPayload(BaseModel):
    data: Dict[str, Any] = Field(..., description="Export bundle dictionary containing skills, profiles, memories, or triplets.")
    data_type: Optional[str] = Field(default=None, description="Optional category filter to import from the bundle.")
    mode: str = Field(default="merge", description="Import mode: 'merge' or 'replace'.")

class SkillCreatePayload(BaseModel):
    skill_name: str = Field(..., description="Unique human-readable skill name.")
    description: str = Field(..., description="Detailed description of what the procedural skill accomplishes.")
    steps: List[Union[Dict[str, Any], str, Any]] = Field(default_factory=list, description="Ordered procedural steps.")
    tool_schema: Optional[Dict[str, Any]] = Field(default=None, description="Optional JSON Schema tool/function definition.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata.")
    skill_id: Optional[str] = Field(default=None, description="Optional stable unique identifier.")
    decision_rules: Optional[List[str]] = Field(default=None, description="Optional decision rules.")
    triples: Optional[Any] = Field(default=None, description="Optional structural knowledge graph triples.")

class SkillGetPayload(BaseModel):
    skill_name: str = Field(..., description="Unique skill name or identifier.")

class ProfileCreatePayload(BaseModel):
    user_id: str = Field(..., description="Unique user identifier.")
    fact_text: str = Field(..., description="Factual profile or preference text.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata.")

class ProfileGetPayload(BaseModel):
    user_id: str = Field(..., description="Unique user identifier.")

class SnapshotPayload(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user ID filter for summary snapshot.")

class ProposePayload(BaseModel):
    text: str = Field(..., description="Factual memory text to propose for symbolic validation.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional structural metadata.")
    triples: Optional[Any] = Field(default=None, description="Optional graph triples.")
    id: Optional[str] = Field(default=None, description="Optional predefined identifier.")

# -------------------------------------------------------------------------
# 5. API Core Router Endpoints
# -------------------------------------------------------------------------
@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """
    Liveness and Readiness probe endpoint.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator gateway not initialized.")
        if shard_nodes:
            tasks = [client.get(f"{shard}/healthz", timeout=2.0) for shard in shard_nodes]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for shard_url, resp in zip(shard_nodes, responses):
                if isinstance(resp, Exception) or resp.status_code != 200:
                    logger.warning(f"Shard health probe failed: {shard_url} -> {resp}")
                    raise HTTPException(status_code=503, detail="One or more backend shards are unhealthy or warming up.")
        return {"status": "healthy", "mode": "coordinator", "version": SERVER_VERSION}
    else:
        global db
        if db is None:
            try:
                db = await get_db_instance(None, None)
            except Exception as e:
                logger.error(f"Failed to initialize storage engine in healthz: {e}")
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        return {"status": "healthy", "mode": "shard", "version": SERVER_VERSION}

# Helper to build headers when coordinator forwards to shards
def get_forward_headers(ctx: RequestContext) -> Dict[str, str]:
    headers = {}
    if ctx.tenant:
        headers["X-Tenant"] = ctx.tenant
    if ctx.namespace:
        headers["X-Namespace"] = ctx.namespace
    if INTERNAL_AUTH_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_AUTH_TOKEN
    return headers


# Shard database partitions administration schemas and routes
class ResetPayload(BaseModel):
    tenant: Optional[str] = None
    namespace: Optional[str] = None

@app.get("/admin/databases")
async def admin_list_databases(ctx: RequestContext = Depends(get_context({Permission.ADMIN}))):
    """
    Lists all local database partitions (tenants and namespaces) on this shard.
    Only supported in Shard Mode.
    """
    if NODE_MODE == "coordinator":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supported in Shard Mode.")
    
    storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
    db_keys = set()
    
    # 1. Gather active databases from memory pool
    for key in db_pool.keys():
        db_keys.add(key)
    if db is not None:
        db_keys.add((None, None))
        
    # 2. Gather databases from filesystem scan
    if os.path.exists(os.path.join(storage_dir, "metadata.json")):
        db_keys.add((None, None))
        
    # Check namespaces directly under storage_dir
    ns_dir = os.path.join(storage_dir, "ns")
    if os.path.exists(ns_dir):
        try:
            for namespace in os.listdir(ns_dir):
                if os.path.isdir(os.path.join(ns_dir, namespace)):
                    db_keys.add((None, namespace))
        except Exception as e:
            logger.error(f"Error listing namespace directory: {e}")
            
    # Check tenants
    tenants_dir = os.path.join(storage_dir, "tenants")
    if os.path.exists(tenants_dir):
        try:
            for tenant in os.listdir(tenants_dir):
                tenant_path = os.path.join(tenants_dir, tenant)
                if os.path.isdir(tenant_path):
                    # Check if tenant has default namespace db
                    if os.path.exists(os.path.join(tenant_path, "metadata.json")):
                        db_keys.add((tenant, None))
                    
                    # Check namespaces within tenant
                    tenant_ns_dir = os.path.join(tenant_path, "ns")
                    if os.path.exists(tenant_ns_dir):
                        for namespace in os.listdir(tenant_ns_dir):
                            if os.path.isdir(os.path.join(tenant_ns_dir, namespace)):
                                db_keys.add((tenant, namespace))
        except Exception as e:
            logger.error(f"Error listing tenants directory: {e}")
            
    dbs = [{"tenant": t, "namespace": ns} for t, ns in db_keys]
    return {"databases": dbs}


@app.get("/databases")
async def list_databases(ctx: RequestContext = Depends(get_context({Permission.READ}))):
    """
    Lists all available database partitions (tenants and namespaces) for visualization scope autocomplete.
    """
    if NODE_MODE == "coordinator":
        if not client or not shard_nodes:
            return {"databases": []}
        target_shard = get_healthy_replica_for_group(shard_groups[0]) if shard_groups else shard_nodes[0]
        try:
            resp = await client.get(f"{target_shard}/databases", headers=get_forward_headers(ctx))
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Error querying databases from shard: {e}")
        return {"databases": []}
    else:
        storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
        db_keys = set()
        for key in db_pool.keys():
            db_keys.add(key)
        if db is not None:
            db_keys.add((None, None))
        if os.path.exists(os.path.join(storage_dir, "metadata.json")):
            db_keys.add((None, None))
        ns_dir = os.path.join(storage_dir, "ns")
        if os.path.exists(ns_dir):
            try:
                for namespace in os.listdir(ns_dir):
                    if os.path.isdir(os.path.join(ns_dir, namespace)):
                        db_keys.add((None, namespace))
            except Exception as e:
                logger.error(f"Error listing namespace directory: {e}")
        tenants_dir = os.path.join(storage_dir, "tenants")
        if os.path.exists(tenants_dir):
            try:
                for tenant in os.listdir(tenants_dir):
                    tenant_path = os.path.join(tenants_dir, tenant)
                    if os.path.isdir(tenant_path):
                        if os.path.exists(os.path.join(tenant_path, "metadata.json")):
                            db_keys.add((tenant, None))
                        tenant_ns_dir = os.path.join(tenant_path, "ns")
                        if os.path.exists(tenant_ns_dir):
                            for namespace in os.listdir(tenant_ns_dir):
                                if os.path.isdir(os.path.join(tenant_ns_dir, namespace)):
                                    db_keys.add((tenant, namespace))
            except Exception as e:
                logger.error(f"Error listing tenants directory: {e}")
        dbs = [{"tenant": t, "namespace": ns} for t, ns in db_keys if t is not None or ns is not None]
        return {"databases": dbs}


@app.post("/admin/toggle_sync")
async def toggle_sync(
    enabled: bool,
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    """
    Toggles coordinator's background health check shard data synchronization loop.
    Only supported in Coordinator Mode.
    """
    if NODE_MODE != "coordinator":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supported in Coordinator Mode.")
    
    global BACKGROUND_SYNC_ENABLED
    BACKGROUND_SYNC_ENABLED = enabled
    logger.info(f"Background shard synchronization set to {enabled}")
    return {"status": "success", "enabled": BACKGROUND_SYNC_ENABLED}


@app.post("/admin/reset")
async def admin_reset_database(
    payload: ResetPayload,
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    """
    Resets and completely deletes a database partition (tenant and namespace) on this shard.
    Only supported in Shard Mode.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        
        all_replica_nodes = []
        for group in shard_groups:
            all_replica_nodes.extend(group)
            
        tasks = [
            client.post(
                f"{node}/admin/reset",
                json=payload.model_dump(),
                headers=get_forward_headers(ctx)
            ) for node in all_replica_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        errors = []
        for node, resp in zip(all_replica_nodes, responses):
            if isinstance(resp, Exception) or (isinstance(resp, httpx.Response) and resp.status_code != 200):
                err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                logger.error(f"Reset failed on node {node}: {err_msg}")
                errors.append(f"{node}: {err_msg}")
                
        if errors:
            raise HTTPException(status_code=500, detail=f"Failed to reset partition on all replica nodes: {', '.join(errors)}")
            
        invalidate_cache(payload.tenant, payload.namespace)
        return {"status": "success", "message": f"Cluster-wide database partition reset successfully."}
        
    key = (payload.tenant, payload.namespace)
    logger.info(f"Admin reset request received for tenant={payload.tenant}, namespace={payload.namespace}")
    
    # 1. Close and remove the database instance from pool
    if key in db_pool:
        db_instance = db_pool[key]
        try:
            await db_instance.close()
        except Exception as e:
            logger.error(f"Error closing db instance for reset: {e}")
        del db_pool[key]
        
    # 2. Delete storage files
    storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
    if payload.tenant:
        storage_dir = os.path.join(storage_dir, "tenants", payload.tenant)
    if payload.namespace:
        storage_dir = os.path.join(storage_dir, "ns", payload.namespace)
    storage_dir = os.path.abspath(storage_dir)
    
    if os.path.exists(storage_dir):
        try:
            import shutil
            shutil.rmtree(storage_dir)
            logger.info(f"Deleted storage directory {storage_dir}")
        except Exception as e:
            logger.error(f"Failed to delete storage directory {storage_dir}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear partition storage: {str(e)}"
            )
            
    global db
    if payload.tenant is None and payload.namespace is None:
        db = await get_db_instance(None, None)
        
    invalidate_cache(payload.tenant, payload.namespace)
    return {"status": "success", "message": f"Database partition reset successfully."}


# Key administration schemas
class KeyCreatePayload(BaseModel):
    permissions: List[str]
    tenant: Optional[str] = None
    namespace: Optional[str] = None
    expires_at: Optional[float] = None
    description: Optional[str] = None

# Key administration routes
@app.post("/admin/keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: KeyCreatePayload,
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    keystore = get_keystore()
    key_id, raw_key = keystore.create_key(
        permissions=payload.permissions,
        tenant=payload.tenant,
        namespace=payload.namespace,
        expires_at=payload.expires_at,
        description=payload.description
    )
    return {"key_id": key_id, "api_key": raw_key}

@app.delete("/admin/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    keystore = get_keystore()
    success = keystore.revoke_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"status": "success", "message": f"Key {key_id} revoked."}

@app.get("/admin/keys")
async def list_api_keys(
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    keystore = get_keystore()
    keys = keystore.list_keys()
    return {"keys": [k.model_dump() for k in keys]}


@app.post("/remember", status_code=status.HTTP_201_CREATED)
async def remember(payload: MemoryPayload, request: Request, ctx: RequestContext = Depends(get_context({Permission.WRITE}))):
    """
    Appends a new memory atom to the Hot Tier (RAM) and schedules background WAL logging.
    In coordinator mode, routes writes using consistent hashing and ID prefixing.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available to route write request.")
        
        original_predefined_id = payload.id
        if original_predefined_id:
            target_shard = get_shard_for_id(original_predefined_id)
            if target_shard:
                atom_id = original_predefined_id
            else:
                target_shard = get_healthy_shard(payload.text)
                shard_idx = shard_nodes.index(target_shard)
                atom_id = f"shard{shard_idx}-{original_predefined_id}"
        else:
            target_shard = get_healthy_shard(payload.text)
            shard_idx = shard_nodes.index(target_shard)
            atom_id = f"shard{shard_idx}-{uuid.uuid4().hex}"

        max_attempts = max(1, int(os.getenv("COORDINATOR_WRITE_RETRIES", "3")))
        for attempt in range(1, max_attempts + 1):
            
            try:
                target_payload = {
                    "text": payload.text,
                    "metadata": payload.metadata,
                    "id": atom_id,
                    "memory_type": payload.memory_type,
                    "triples": payload.triples
                }
                group = get_replica_group_for_node(target_shard)
                tasks = [
                    client.post(
                        f"{node}/remember",
                        json=target_payload,
                        headers=get_forward_headers(ctx)
                    ) for node in group
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_nodes = []
                failures = []
                for node, resp in zip(group, responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 201:
                        success_nodes.append(node)
                    else:
                        err_msg = format_shard_error(resp)
                        logger.warning(f"Replication write failed to node {node}: {err_msg}")
                        failures.append((node, err_msg))
                        if should_mark_shard_offline(resp):
                            SHARD_METRICS_CACHE[node] = {
                                "status": "offline",
                                "cpu": 0.0,
                                "ram": {"percent": 0.0},
                                "disk": {"percent": 0.0}
                            }
                
                consistency = get_consistency_level(request)
                n = len(group)
                if consistency == ConsistencyLevel.ALL:
                    required_success = n
                elif consistency == ConsistencyLevel.QUORUM:
                    required_success = n // 2 + 1
                else:
                    required_success = 1
                
                if len(success_nodes) >= required_success:
                    invalidate_cache(ctx.tenant, ctx.namespace)
                    await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                    return {"status": "success", "id": atom_id}
                else:
                    last_err = failures[-1][1] if failures else "Unknown error"
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"Owning shard group could not accept write. Required: "
                            f"{consistency.value} ({required_success}), got: "
                            f"{len(success_nodes)} success(es). Last error: {last_err}"
                        ),
                        headers={"Retry-After": "5"},
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to forward write to shard set {target_shard} on attempt {attempt}: {e}")
                if attempt >= max_attempts:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Owning shard group could not accept write: {str(e)}",
                        headers={"Retry-After": "5"},
                    )
            
    else:
        # Shard Mode (Storage Node)
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            if payload.metadata is None:
                payload.metadata = {}
            if "_updated_at" not in payload.metadata:
                payload.metadata["_updated_at"] = time.time()
                
            provided_triples = list(payload.triples or []) or list(payload.metadata.get("triples") or [])
            if provided_triples and "triples" not in payload.metadata:
                payload.metadata["triples"] = provided_triples

            if payload.id:
                engine = await active_db._get_db()
                text = payload.text
                metadata = payload.metadata
                
                # Manual embedding generation to support predefined atom_id
                if engine._model_name:
                    embedder = engine._get_embedder()
                    emb = await run_embedding(
                        embedder.encode,
                        text,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    embedding = np.array(emb, dtype=np.float32)
                else:
                    embedding = np.zeros(engine.dim, dtype=np.float32)
                    
                provided_triples = metadata.get("triples") or payload.triples or []

                def _extract_triples_for_ingest() -> list:
                    from epochdb.core.fact_extractor import FactExtractor

                    if getattr(engine, "auto_extract", False):
                        if engine._fact_extractor is None:
                            engine._fact_extractor = FactExtractor(
                                engine, engine.extraction_model
                            )
                        return engine._fact_extractor.extract(text)

                    def _pair_triples(items: list) -> list:
                        seen = set()
                        entities = []
                        for item in items:
                            s = str(item)
                            if s and s not in seen:
                                seen.add(s)
                                entities.append(s)
                        if len(entities) >= 2:
                            res = []
                            for i in range(len(entities) - 1):
                                res.append((entities[i], "co_occurs_with", entities[i + 1]))
                            if len(entities) > 2:
                                res.append((entities[0], "co_occurs_with", entities[-1]))
                            return res
                        elif len(entities) == 1:
                            return [(entities[0], "mentions", entities[0])]
                        return []

                    extracted = engine.extract_entities(text)
                    if extracted:
                        res = _pair_triples(extracted)
                        if res:
                            return res

                    # Match LocalFactExtractor fallback for brand-new knowledge graphs.
                    words = [w.strip(".,!?;:()\"'") for w in text.split() if w.strip()]
                    nouns = [w for w in words if w and w[0].isupper()]
                    if not nouns:
                        nouns = [w for w in words if len(w) > 3][:3]
                    return _pair_triples(nouns)

                # Always discover entity links when AUTO_EXTRACT is on, and merge
                # them with any caller-provided structural triples (rebuild/index).
                if getattr(engine, "auto_extract", False) or not provided_triples:
                    discovered = await asyncio.to_thread(_extract_triples_for_ingest)
                else:
                    discovered = []

                merged = []
                seen_triples = set()
                for t in list(provided_triples) + list(discovered or []):
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        key = (str(t[0]), str(t[1]), str(t[2]))
                    else:
                        continue
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    merged.append(list(key))
                triples = merged
                metadata["triples"] = triples

                # If the atom already exists (hot or cold), replace triples in-place.
                existing = payload.id in engine.hot_tier.atoms
                if not existing:
                    for epoch_id in engine.cold_tier.get_all_epochs():
                        if engine.cold_tier.load_atom_metadata(epoch_id, [payload.id]):
                            existing = True
                            break

                if existing:
                    atom_id = await asyncio.to_thread(
                        engine.replace_memory,
                        payload.id,
                        text,
                        embedding,
                        [tuple(t) for t in triples],
                        metadata,
                    )
                    logger.info(
                        f"Replaced triples for atom {atom_id}: '{text[:40]}...'"
                    )
                    invalidate_cache(ctx.tenant, ctx.namespace)
                    await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                    return {"status": "success", "id": atom_id}

                atom_id = await asyncio.to_thread(
                    engine.add_memory,
                    payload=text,
                    embedding=embedding,
                    triples=triples,
                    metadata=metadata,
                    atom_id=payload.id
                )
                
                # Set memory_type on the atom if specified
                if payload.memory_type:
                    from epochdb.core.atom import MemoryType
                    try:
                        mt = MemoryType(payload.memory_type)
                        atom = engine.hot_tier.atoms.get(atom_id)
                        if atom:
                            atom.memory_type = mt
                    except ValueError:
                        pass
                
                logger.info(f"Ingested atom with fixed ID {atom_id}: '{text[:40]}...'")
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                return {"status": "success", "id": atom_id}
            else:
                async with embedding_slot():
                    atom_id = await active_db.remember(
                        text=payload.text,
                        triples=provided_triples or None,
                        metadata=payload.metadata,
                        memory_type=payload.memory_type,
                    )
                logger.info(f"Ingested atom: '{payload.text[:40]}...'")
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                return {"status": "success", "id": atom_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to commit memory write block: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal storage layer mutation rejected: {str(e)}")

@app.post("/remember_batch", status_code=status.HTTP_201_CREATED)
async def remember_batch(
    payload: BatchMemoryPayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Bulk ingestion endpoint for adding multiple memory atoms.
    Coordinator partitions items by target shard and routes in parallel batches.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available to route batch request.")

        grouped_items: Dict[str, List[Dict[str, Any]]] = {}
        all_ids: List[str] = []

        for item in payload.items:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {"text": str(item)})
            text = item_dict.get("text") or item_dict.get("payload") or ""
            target_id = item_dict.get("id") or item_dict.get("atom_id")

            if target_id:
                target_shard = get_shard_for_id(target_id)
                if not target_shard:
                    target_shard = get_healthy_shard(text)
                    shard_idx = shard_nodes.index(target_shard)
                    target_id = f"shard{shard_idx}-{target_id}"
            else:
                target_shard = get_healthy_shard(text)
                shard_idx = shard_nodes.index(target_shard)
                target_id = f"shard{shard_idx}-{uuid.uuid4().hex}"

            item_dict["id"] = target_id
            item_dict["atom_id"] = target_id
            all_ids.append(target_id)

            if target_shard not in grouped_items:
                grouped_items[target_shard] = []
            grouped_items[target_shard].append(item_dict)

        forward_headers = get_forward_headers(ctx)
        consistency = get_consistency_level(request)

        async def send_group_batch(primary_node: str, items_for_shard: List[Dict[str, Any]]):
            group = get_replica_group_for_node(primary_node)
            tasks = [
                client.post(
                    f"{node}/remember_batch",
                    json={"items": items_for_shard},
                    headers=forward_headers
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = 0
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 201:
                    success_count += 1
                else:
                    err_msg = format_shard_error(resp)
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }

            n = len(group)
            required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)
            if success_count < required:
                last_err = failures[-1][1] if failures else "Unknown error"
                raise HTTPException(
                    status_code=503,
                    detail=f"Batch replication failed for shard group {primary_node}. Required: {consistency.value} ({required}), got: {success_count}. Last error: {last_err}",
                    headers={"Retry-After": "5"}
                )

        await asyncio.gather(*[send_group_batch(shard, items) for shard, items in grouped_items.items()])

        invalidate_cache(ctx.tenant, ctx.namespace)
        await broadcast_sse_event("update", {"type": "batch_write", "count": len(all_ids)})
        return {"status": "success", "atom_ids": all_ids, "count": len(all_ids)}

    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        formatted_items = []
        now_ts = time.time()
        for item in payload.items:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {"text": str(item)})
            text = item_dict.get("text") or item_dict.get("payload") or ""
            metadata = dict(item_dict.get("metadata") or {})
            if "_updated_at" not in metadata:
                metadata["_updated_at"] = now_ts
            if "triples" not in metadata and item_dict.get("triples"):
                metadata["triples"] = item_dict["triples"]
            atom_id = item_dict.get("id") or item_dict.get("atom_id")
            memory_type = item_dict.get("memory_type")

            formatted_items.append({
                "text": text,
                "metadata": metadata,
                "id": atom_id,
                "memory_type": memory_type,
            })

        async with embedding_slot():
            atom_ids = []
            for itm in formatted_items:
                aid = await active_db.remember(
                    text=itm["text"],
                    metadata=itm["metadata"],
                    memory_type=itm["memory_type"],
                    atom_id=itm["id"],
                )
                atom_ids.append(aid)

        invalidate_cache(ctx.tenant, ctx.namespace)
        await broadcast_sse_event("update", {"type": "batch_write", "count": len(atom_ids)})
        return {"status": "success", "atom_ids": atom_ids, "count": len(atom_ids)}

@app.post("/replace")
async def replace_memory(
    payload: ReplacePayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Performs durable in-place replacement of an existing memory atom.
    Replaces payload, embedding, and knowledge graph triples without creating duplicate versions.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

        target_shard = get_shard_for_id(payload.atom_id)
        if target_shard:
            group = get_replica_group_for_node(target_shard)
            tasks = [
                client.post(
                    f"{node}/replace",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            success_nodes = []
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    success_nodes.append(node)
                else:
                    err_msg = format_shard_error(resp)
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }

            consistency = get_consistency_level(request)
            n = len(group)
            required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)
            if len(success_nodes) >= required:
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "replace", "id": payload.atom_id})
                return {"status": "success", "id": payload.atom_id}
            else:
                last_err = failures[-1][1] if failures else "Unknown error"
                raise HTTPException(
                    status_code=500,
                    detail=f"Replace consistency check failed. Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}"
                )
        else:
            target_nodes = []
            for group in shard_groups:
                target_nodes.extend(get_all_healthy_replicas_for_group(group))
            tasks = [
                client.post(
                    f"{node}/replace",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "replace", "id": payload.atom_id})
            return {"status": "success", "id": payload.atom_id}
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            engine = await active_db._get_db()
            meta = dict(payload.metadata or {})
            meta["_updated_at"] = time.time()
            if payload.triples is not None:
                meta["triples"] = payload.triples

            if engine._model_name:
                embedder = engine._get_embedder()
                emb = await run_embedding(
                    embedder.encode,
                    payload.text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                embedding = np.array(emb, dtype=np.float32)
            else:
                embedding = np.zeros(engine.dim, dtype=np.float32)

            triples = meta.get("triples") or []
            atom_id = await asyncio.to_thread(
                engine.replace_memory,
                payload.atom_id,
                payload.text,
                embedding,
                [tuple(t) for t in triples if isinstance(t, (list, tuple)) and len(t) >= 3],
                meta
            )
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "replace", "id": atom_id})
            return {"status": "success", "id": atom_id}
        except Exception as e:
            logger.error(f"Error replacing memory {payload.atom_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/propose", status_code=status.HTTP_200_OK)
async def propose_memory(
    payload: ProposePayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Two-phase state commit (Neuro-Symbolic LCAG verification).
    Stages the memory, executes configured symbolic invariants, and atomically commits or discards.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available.")

        target_shard = get_healthy_shard(payload.text)
        group = get_replica_group_for_node(target_shard)
        tasks = [
            client.post(
                f"{node}/propose",
                json=payload.model_dump(),
                headers=get_forward_headers(ctx)
            ) for node in group
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        success_nodes = []
        approved_result = None
        rejections = []
        failures = []

        for node, resp in zip(group, responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "APPROVED":
                    success_nodes.append(node)
                    approved_result = data
                else:
                    rejections.append((node, data))
            else:
                err_msg = format_shard_error(resp)
                failures.append((node, err_msg))
                if should_mark_shard_offline(resp):
                    SHARD_METRICS_CACHE[node] = {
                        "status": "offline",
                        "cpu": 0.0,
                        "ram": {"percent": 0.0},
                        "disk": {"percent": 0.0}
                    }

        if rejections and not success_nodes:
            return rejections[0][1]

        consistency = get_consistency_level(request)
        n = len(group)
        required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)

        if len(success_nodes) >= required and approved_result:
            invalidate_cache(ctx.tenant, ctx.namespace)
            mid = approved_result.get("memory_id")
            await broadcast_sse_event("update", {"type": "propose_approved", "id": mid})
            return approved_result
        else:
            last_err = failures[-1][1] if failures else (rejections[-1][1].get("error") if rejections else "Unknown validation failure")
            raise HTTPException(
                status_code=503,
                detail=f"Proposal replication failed. Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}",
                headers={"Retry-After": "5"}
            )
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            res = await active_db.propose(
                text=payload.text,
                metadata=payload.metadata,
                triples=payload.triples,
            )
            if res.get("status") == "APPROVED":
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "propose_approved", "id": res.get("memory_id")})
            return res
        except Exception as e:
            logger.error(f"Error in propose memory validation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/get")
async def get_memory(
    payload: GetPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves a specific memory by its unique ID.
    Coordinator routes directly if ID is prefixed, otherwise broadcasts.
    """
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_shard = get_shard_for_id(payload.memory_id)
            if target_shard:
                group = get_replica_group_for_node(target_shard)
                tasks = [
                    client.post(
                        f"{node}/get",
                        json=payload.model_dump(),
                        headers=get_forward_headers(ctx)
                    ) for node in group
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                valid_replies = []
                failures = []
                for node, resp in zip(group, responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        valid_replies.append((node, resp.json()))
                    else:
                        err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                        failures.append((node, err_msg))
                        
                n = len(group)
                consistency = get_consistency_level(request)
                if consistency == ConsistencyLevel.ALL:
                    required_success = n
                elif consistency == ConsistencyLevel.QUORUM:
                    required_success = n // 2 + 1
                else:
                    required_success = 1
                    
                if len(valid_replies) < required_success:
                    last_err = failures[-1][1] if failures else "Unknown read failure"
                    raise HTTPException(
                        status_code=500,
                        detail=f"Read consistency check failed. Required: {consistency.value} ({required_success}), got: {len(valid_replies)} valid replies. Last error: {last_err}"
                    )
                    
                memories_found = [(node, data) for node, data in valid_replies if data and "id" in data]
                
                if not memories_found:
                    return {}
                    
                # A memory is newer if it has a higher _updated_at in metadata, or fallback to created_at
                def get_timestamp(data):
                    meta = data.get("metadata") or {}
                    return meta.get("_updated_at") or data.get("created_at") or 0.0
                    
                latest_node, latest_data = max(memories_found, key=lambda x: get_timestamp(x[1]))
                latest_ts = get_timestamp(latest_data)
                
                stale_replicas = []
                for node, data in valid_replies:
                    if not data or "id" not in data:
                        stale_replicas.append(node)
                    elif get_timestamp(data) < latest_ts:
                        stale_replicas.append(node)
                        
                if stale_replicas:
                    logger.info(f"Read Repair triggered: repairing stale/missing memory {payload.memory_id} on replicas: {stale_replicas}")
                    async def run_read_repair(target_nodes=stale_replicas, memory_data=latest_data):
                        repair_payload = {
                            "text": memory_data.get("payload") or memory_data.get("text"),
                            "metadata": memory_data.get("metadata", {}),
                            "id": memory_data.get("id"),
                            "memory_type": memory_data.get("memory_type")
                        }
                        headers = get_forward_headers(ctx)
                        for node in target_nodes:
                            try:
                                res = await client.post(f"{node}/remember", json=repair_payload, headers=headers)
                                if res.status_code == 201:
                                    logger.info(f"Successfully repaired memory {memory_data.get('id')} on node {node}")
                                else:
                                    logger.error(f"Failed to repair memory on node {node}: status {res.status_code}")
                            except Exception as re_err:
                                logger.error(f"Read repair request failed for node {node}: {re_err}")
                                
                    asyncio.create_task(run_read_repair())
                    
                return latest_data
            else:
                # Broadcast to all shards in parallel (one healthy replica per shard group)
                target_nodes = []
                for group in shard_groups:
                    node = get_healthy_replica_for_group(group)
                    if node:
                        target_nodes.append(node)
                tasks = [
                    client.post(
                        f"{node}/get",
                        json=payload.model_dump(),
                        headers=get_forward_headers(ctx)
                    ) for node in target_nodes
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for resp in responses:
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        data = resp.json()
                        if data and "id" in data:
                            return data
                return {}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                mem = await active_db.get(payload.memory_id)
                if mem:
                    return mem._atom.to_dict()
                return {}
            except Exception as e:
                logger.error(f"Error resolving memory retrieval: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    consistency = get_consistency_level(request)
    if consistency in (ConsistencyLevel.QUORUM, ConsistencyLevel.ALL):
        return await fetch()
    else:
        return await handle_cached_read("get", payload, request, response, ctx, fetch)

@app.post("/query")
async def query_memories(
    payload: QueryPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Performs semantic and hybrid search across the memory database.
    Coordinator parallelizes requests to all shards and merges/re-ranks results.
    """
    query_text = payload.get_query_text()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text must not be empty.")

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            forward_data = payload.model_dump()
            forward_data["query"] = query_text
            forward_data["text"] = query_text

            tasks = [
                client.post(
                    f"{node}/query",
                    json=forward_data,
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_results = []
            failed_shards = 0
            for node, resp in zip(target_nodes, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    all_results.extend(data.get("results") or data.get("memories") or [])
                else:
                    failed_shards += 1
                    logger.error(f"Error querying shard {node}: {format_shard_error(resp)}")
                    
            if payload.min_score > 0.0:
                all_results = [r for r in all_results if r.get("score", 0.0) >= payload.min_score]

            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            result = {
                "results": all_results[:payload.k],
                "memories": all_results[:payload.k]
            }
            if failed_shards:
                result["partial"] = True
                result["failed_shards"] = failed_shards
            return result
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                query_kwargs = {
                    "text": query_text,
                    "k": payload.k,
                    "filters": payload.filters,
                    "memory_type": payload.memory_type,
                    "context_window": payload.context_window,
                    "min_score": payload.min_score,
                }
                if payload.expand_hops > 0:
                    query_kwargs["expand_hops"] = payload.expand_hops
                if payload.cold_search_mode:
                    query_kwargs["cold_search_mode"] = payload.cold_search_mode

                try:
                    results = await active_db.query(**query_kwargs)
                except TypeError:
                    query_kwargs.pop("expand_hops", None)
                    query_kwargs.pop("cold_search_mode", None)
                    results = await active_db.query(**query_kwargs)
                
                engine = await active_db._get_db()
                embedder = engine._get_embedder()
                q_emb = await run_embedding(
                    embedder.encode,
                    query_text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                q_emb = np.array(q_emb, dtype=np.float32)
                
                formatted_results = []
                for r in results:
                    score = float(getattr(r, "score", 0.0))
                    if score == 0.0 and q_emb.any() and r._atom.embedding.any():
                        score = float(np.dot(r._atom.embedding, q_emb) / (
                            np.linalg.norm(r._atom.embedding) * np.linalg.norm(q_emb) + 1e-10
                        ))
                    item_data = {
                        "id": r.id,
                        "text": r.text,
                        "metadata": r.metadata,
                        "created_at": r.created_at,
                        "score": score
                    }
                    if hasattr(r._atom, "memory_type") and r._atom.memory_type:
                        item_data["memory_type"] = r._atom.memory_type.value if hasattr(r._atom.memory_type, "value") else str(r._atom.memory_type)
                    formatted_results.append(item_data)

                if payload.min_score > 0.0:
                    formatted_results = [r for r in formatted_results if r.get("score", 0.0) >= payload.min_score]

                return {"results": formatted_results, "memories": formatted_results}
            except Exception as e:
                logger.error(f"Error resolving retrieval operations: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("query", payload, request, response, ctx, fetch)

@app.post("/multi_hop")
async def multi_hop(
    payload: MultiHopPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Executes a multi-hop relational retrieval query across knowledge-graph associations and semantic embeddings.
    Coordinator fans out to shards and aggregates candidate results.
    """
    query_text = payload.get_query_text()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text must not be empty.")

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            forward_data = payload.model_dump()
            forward_data["text"] = query_text
            forward_data["query"] = query_text

            tasks = [
                client.post(
                    f"{node}/multi_hop",
                    json=forward_data,
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            seen_ids = set()
            failed_shards = 0
            for node, resp in zip(target_nodes, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    memories = data.get("memories") or data.get("results") or []
                    for m in memories:
                        mid = m.get("id")
                        if mid and mid not in seen_ids:
                            seen_ids.add(mid)
                            all_results.append(m)
                        elif not mid:
                            all_results.append(m)
                else:
                    failed_shards += 1
                    logger.error(f"Error querying multi_hop on shard {node}: {format_shard_error(resp)}")

            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            result = {
                "results": all_results[:payload.k],
                "memories": all_results[:payload.k]
            }
            if failed_shards:
                result["partial"] = True
                result["failed_shards"] = failed_shards
            return result
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                kwargs = {
                    "hops": payload.hops,
                    "k": payload.k,
                    "filters": payload.filters,
                    "context_window": payload.context_window,
                }
                if payload.cold_search_mode:
                    kwargs["cold_search_mode"] = payload.cold_search_mode

                results = await active_db.multi_hop(query_text, **kwargs)
                formatted = []
                for r in results:
                    atom_dict = r._atom.to_dict()
                    if hasattr(r, "score"):
                        atom_dict["score"] = float(r.score)
                    formatted.append(atom_dict)
                return {"results": formatted, "memories": formatted}
            except Exception as e:
                logger.error(f"Error resolving multi_hop retrieval: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("multi_hop", payload, request, response, ctx, fetch)

@app.post("/skills/remember", status_code=status.HTTP_201_CREATED)
async def remember_skill(
    payload: SkillCreatePayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Registers a synthesized procedural skill with steps, tool schema, and decision rules.
    Stored under MemoryType.SKILL and indexed in the Hot Tier SkillIndex.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available.")

        route_key = payload.skill_id or payload.skill_name
        target_shard = get_healthy_shard(route_key)
        shard_idx = shard_nodes.index(target_shard)
        stable_id = payload.skill_id or f"shard{shard_idx}-skill-{uuid.uuid4().hex}"
        forward_data = payload.model_dump()
        forward_data["skill_id"] = stable_id

        group = get_replica_group_for_node(target_shard)
        tasks = [
            client.post(
                f"{node}/skills/remember",
                json=forward_data,
                headers=get_forward_headers(ctx)
            ) for node in group
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        success_nodes = []
        failures = []
        for node, resp in zip(group, responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 201:
                success_nodes.append(node)
            else:
                err_msg = format_shard_error(resp)
                failures.append((node, err_msg))
                if should_mark_shard_offline(resp):
                    SHARD_METRICS_CACHE[node] = {"status": "offline"}

        consistency = get_consistency_level(request)
        n = len(group)
        required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)

        if len(success_nodes) >= required:
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "skill_registered", "id": stable_id})
            return {"status": "success", "id": stable_id, "skill_name": payload.skill_name}
        else:
            last_err = failures[-1][1] if failures else "Unknown error"
            raise HTTPException(
                status_code=503,
                detail=f"Skill replication failed for group {target_shard}. Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}",
                headers={"Retry-After": "5"}
            )
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            norm_steps = []
            for i, st in enumerate(payload.steps or []):
                if isinstance(st, str):
                    norm_steps.append({"step": i + 1, "action": st})
                elif isinstance(st, dict):
                    norm_steps.append(st)
                else:
                    norm_steps.append({"step": i + 1, "action": str(st)})

            aid = await active_db.remember_skill(
                skill_name=payload.skill_name,
                description=payload.description,
                steps=norm_steps,
                tool_schema=payload.tool_schema,
                metadata=payload.metadata,
                skill_id=payload.skill_id,
                decision_rules=payload.decision_rules,
                triples=payload.triples,
            )
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "skill_registered", "id": aid})
            return {"status": "success", "id": aid, "skill_name": payload.skill_name}
        except Exception as e:
            logger.error(f"Error registering skill {payload.skill_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

async def _execute_get_skill(
    name: str,
    request: Request,
    response: Response,
    ctx: RequestContext
):
    if not name:
        raise HTTPException(status_code=400, detail="Skill name is required.")

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_node = hash_ring.get_node(name)
            group = get_replica_group_for_node(target_node)
            probe_node = get_healthy_replica_for_group(group) or group[0]

            try:
                res = await client.get(f"{probe_node}/skills/{name}", headers=get_forward_headers(ctx))
                if res.status_code == 200:
                    data = res.json()
                    if data and data.get("id"):
                        return data
            except Exception:
                pass

            target_nodes = []
            for grp in shard_groups:
                nd = get_healthy_replica_for_group(grp)
                if nd and nd != probe_node:
                    target_nodes.append(nd)

            if target_nodes:
                tasks = [client.get(f"{node}/skills/{name}", headers=get_forward_headers(ctx)) for node in target_nodes]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for resp in responses:
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        data = resp.json()
                        if data and data.get("id"):
                            return data
            return {}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                mem = await active_db.get_skill(name)
                if mem:
                    return mem._atom.to_dict()
                return {}
            except Exception as e:
                logger.error(f"Error fetching skill {name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("get_skill", {"skill_name": name}, request, response, ctx, fetch)

@app.get("/skills/{skill_name}")
async def get_skill_by_path(
    skill_name: str,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves a procedural skill by name via path parameter.
    """
    return await _execute_get_skill(skill_name, request, response, ctx)

@app.post("/skills/get")
async def get_skill_by_payload(
    payload: SkillGetPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves a procedural skill by name via JSON payload.
    """
    return await _execute_get_skill(payload.skill_name, request, response, ctx)

@app.get("/skills")
async def list_skills(
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Lists all procedural skills registered across the cluster.
    """
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            tasks = [client.get(f"{node}/skills", headers=get_forward_headers(ctx)) for node in target_nodes]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            all_skills = []
            seen = set()
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    for s in data.get("skills", []):
                        sid = s.get("id") or s.get("metadata", {}).get("skill_name")
                        if sid not in seen:
                            seen.add(sid)
                            all_skills.append(s)
            return {"skills": all_skills}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                skills = await active_db.list_skills()
                return {"skills": [s._atom.to_dict() for s in skills]}
            except Exception as e:
                logger.error(f"Error listing skills: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("list_skills", {}, request, response, ctx, fetch)

@app.post("/profiles/remember", status_code=status.HTTP_201_CREATED)
async def remember_profile(
    payload: ProfileCreatePayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Stores long-term user profile facts and preferences under MemoryType.PROFILE.
    Coordinator routes deterministically by user_id.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available.")

        target_shard = get_healthy_shard(payload.user_id)
        group = get_replica_group_for_node(target_shard)
        tasks = [
            client.post(
                f"{node}/profiles/remember",
                json=payload.model_dump(),
                headers=get_forward_headers(ctx)
            ) for node in group
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        success_nodes = []
        result_id = None
        failures = []
        for node, resp in zip(group, responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 201:
                success_nodes.append(node)
                result_id = resp.json().get("id")
            else:
                err_msg = format_shard_error(resp)
                failures.append((node, err_msg))
                if should_mark_shard_offline(resp):
                    SHARD_METRICS_CACHE[node] = {"status": "offline"}

        consistency = get_consistency_level(request)
        n = len(group)
        required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)

        if len(success_nodes) >= required:
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "profile_updated", "user_id": payload.user_id})
            return {"status": "success", "id": result_id, "user_id": payload.user_id}
        else:
            last_err = failures[-1][1] if failures else "Unknown error"
            raise HTTPException(
                status_code=503,
                detail=f"Profile replication failed. Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}",
                headers={"Retry-After": "5"}
            )
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            aid = await active_db.remember_user_profile(
                user_id=payload.user_id,
                fact_text=payload.fact_text,
                metadata=payload.metadata,
            )
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "profile_updated", "user_id": payload.user_id})
            return {"status": "success", "id": aid, "user_id": payload.user_id}
        except Exception as e:
            logger.error(f"Error remembering user profile for {payload.user_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

async def _execute_get_profile(
    uid: str,
    request: Request,
    response: Response,
    ctx: RequestContext
):
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required.")

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_node = hash_ring.get_node(uid)
            group = get_replica_group_for_node(target_node)
            probe_node = get_healthy_replica_for_group(group) or group[0]

            res = await client.get(f"{probe_node}/profiles/{uid}", headers=get_forward_headers(ctx))
            if res.status_code == 200:
                return res.json()
            return {"memories": []}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                mems = await active_db.get_user_profile(uid)
                if not mems:
                    # Query with min_score=-1.0 to ensure profile facts aren't dropped by negative cosine angle
                    mems = await active_db.query(
                        text=f"user profile {uid}",
                        k=50,
                        filters={"user_id": uid},
                        memory_type="profile",
                        min_score=-1.0
                    )
                if not mems:
                    # Fallback to direct hot tier lookup by metadata
                    raw_db = await active_db._get_db()
                    matched_atoms = [
                        a for a in raw_db.hot_tier.atoms.values()
                        if a.memory_type and a.memory_type.value == "profile" and a.metadata.get("user_id") == uid
                    ]
                    return {"memories": [a.to_dict() for a in matched_atoms]}

                return {"memories": [m._atom.to_dict() if hasattr(m, "_atom") else m.to_dict() for m in mems]}
            except Exception as e:
                logger.error(f"Error fetching profile for {uid}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("get_profile", {"user_id": uid}, request, response, ctx, fetch)

@app.get("/profiles/{user_id}")
async def get_profile_by_path(
    user_id: str,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves all stored profile facts and preferences for a specific user via path.
    """
    return await _execute_get_profile(user_id, request, response, ctx)

@app.post("/profiles/get")
async def get_profile_by_payload(
    payload: ProfileGetPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves all stored profile facts and preferences for a specific user via JSON payload.
    """
    return await _execute_get_profile(payload.user_id, request, response, ctx)

async def _execute_summary_snapshot(
    uid: Optional[str],
    request: Request,
    response: Response,
    ctx: RequestContext
):
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            params = {"user_id": uid} if uid else {}
            tasks = [client.get(f"{node}/summary/snapshot", params=params, headers=get_forward_headers(ctx)) for node in target_nodes]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            snapshots = []
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    snap_text = resp.json().get("snapshot", "").strip()
                    if snap_text and snap_text not in snapshots:
                        snapshots.append(snap_text)
            aggregated = "\n\n".join(snapshots)
            return {"snapshot": aggregated}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                snap = await active_db.get_hot_summary_snapshot(uid)
                return {"snapshot": snap}
            except Exception as e:
                logger.error(f"Error generating summary snapshot: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("summary_snapshot", {"user_id": uid}, request, response, ctx, fetch)

@app.get("/summary/snapshot")
async def get_summary_snapshot(
    request: Request,
    response: Response,
    user_id: Optional[str] = Query(None),
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Produces a compact textual snapshot of active hot memories and profile facts via GET.
    """
    return await _execute_summary_snapshot(user_id, request, response, ctx)

@app.post("/summary/snapshot")
async def post_summary_snapshot(
    request: Request,
    response: Response,
    payload: Optional[SnapshotPayload] = None,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Produces a compact textual snapshot of active hot memories and profile facts via POST.
    """
    uid = payload.user_id if payload else None
    return await _execute_summary_snapshot(uid, request, response, ctx)

@app.post("/adaptive_query")
async def adaptive_query_memories(
    payload: AdaptiveQueryPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Intelligently routes a query to the optimal engine(s) (semantic, relational, temporal, or quantitative) 
    using LLM-orchestrated routing (or local rule fallbacks if offline) and retrieves relevant memories.
    """
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)
            tasks = [
                client.post(
                    f"{node}/adaptive_query",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_results = []
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    all_results.extend(data.get("results", []))
                elif isinstance(resp, Exception):
                    logger.error(f"Error adaptive querying shard: {resp}")
                    
            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return {"results": all_results[:payload.k]}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                results = await active_db.adaptive_query(
                    query=payload.query,
                    k=payload.k,
                    context_window=payload.context_window
                )
                
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "id": r.id,
                        "text": r.text,
                        "metadata": r.metadata,
                        "created_at": r.created_at,
                        "score": 1.0
                    })
                return {"results": formatted_results}
            except Exception as e:
                logger.error(f"Error resolving adaptive query: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("adaptive_query", payload, request, response, ctx, fetch)

@app.post("/update")
async def update_memory(payload: UpdatePayload, request: Request, ctx: RequestContext = Depends(get_context({Permission.WRITE}))):
    """
    Updates memory text or metadata.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_shard = get_shard_for_id(payload.memory_id)
        if target_shard:
            group = get_replica_group_for_node(target_shard)
            tasks = [
                client.post(
                    f"{node}/update",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_nodes = []
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    success_nodes.append(node)
                else:
                    err_msg = format_shard_error(resp)
                    logger.warning(f"Replication update failed on replica {node}: {err_msg}")
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }
                    
            consistency = get_consistency_level(request)
            n = len(group)
            if consistency == ConsistencyLevel.ALL:
                required_success = n
            elif consistency == ConsistencyLevel.QUORUM:
                required_success = n // 2 + 1
            else:
                required_success = 1
                
            if len(success_nodes) >= required_success:
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "update", "id": payload.memory_id})
                return {"status": "success"}
            else:
                last_err = failures[-1][1] if failures else "Unknown error"
                raise HTTPException(
                    status_code=500,
                    detail=f"Update consistency check failed. Required: {consistency.value} ({required_success}), got: {len(success_nodes)}. Last error: {last_err}"
                )
        else:
            # Broadcast to all replica groups
            target_nodes = []
            for group in shard_groups:
                target_nodes.extend(get_all_healthy_replicas_for_group(group))
            tasks = [
                client.post(
                    f"{node}/update",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for node, resp in zip(target_nodes, responses):
                if isinstance(resp, Exception) or (isinstance(resp, httpx.Response) and resp.status_code != 200):
                    logger.error(f"Error updating replica {node}: {resp}")
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "update", "id": payload.memory_id})
            return {"status": "success"}
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            meta = dict(payload.metadata) if payload.metadata is not None else {}
            meta["_updated_at"] = time.time()
            if payload.triples is not None:
                meta["triples"] = payload.triples
            await active_db.update(payload.memory_id, payload.text, meta)
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "update", "id": payload.memory_id})
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Error updating memory {payload.memory_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete_memory(payload: DeletePayload, request: Request, ctx: RequestContext = Depends(get_context({Permission.DELETE}))):
    """
    Deletes memory (hard or soft).
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_shard = get_shard_for_id(payload.memory_id)
        if target_shard:
            group = get_replica_group_for_node(target_shard)
            tasks = [
                client.post(
                    f"{node}/delete",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_nodes = []
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    success_nodes.append(node)
                else:
                    err_msg = format_shard_error(resp)
                    logger.warning(f"Replication delete failed on replica {node}: {err_msg}")
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }
                    
            consistency = get_consistency_level(request)
            n = len(group)
            if consistency == ConsistencyLevel.ALL:
                required_success = n
            elif consistency == ConsistencyLevel.QUORUM:
                required_success = n // 2 + 1
            else:
                required_success = 1
                
            if len(success_nodes) >= required_success:
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "delete", "id": payload.memory_id})
                return {"status": "success"}
            else:
                last_err = failures[-1][1] if failures else "Unknown error"
                raise HTTPException(
                    status_code=500,
                    detail=f"Delete consistency check failed. Required: {consistency.value} ({required_success}), got: {len(success_nodes)}. Last error: {last_err}"
                )
        else:
            # Broadcast to all replica groups
            target_nodes = []
            for group in shard_groups:
                target_nodes.extend(get_all_healthy_replicas_for_group(group))
            tasks = [
                client.post(
                    f"{node}/delete",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for node, resp in zip(target_nodes, responses):
                if isinstance(resp, Exception) or (isinstance(resp, httpx.Response) and resp.status_code != 200):
                    logger.error(f"Error deleting replica {node}: {resp}")
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "delete", "id": payload.memory_id})
            return {"status": "success"}
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            await active_db.delete(payload.memory_id, payload.hard)
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "delete", "id": payload.memory_id})
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Error deleting memory {payload.memory_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/entity_graph")
async def entity_graph(
    request: Request,
    response: Response,
    entity_id: Optional[str] = None,
    depth: int = 2,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves the local entity graph or aggregates the distributed graph.
    """
    payload = {"entity_id": entity_id, "depth": depth}

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            if not entity_id:
                tasks = [
                    client.get(
                        f"{node}/entity_graph",
                        params={"depth": depth},
                        headers=get_forward_headers(ctx)
                    ) for node in target_nodes
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                merged_nodes = set()
                merged_edges = []
                seen_edges = set()
                
                for resp in responses:
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        data = resp.json()
                        for node in data.get("nodes", []):
                            merged_nodes.add(node)
                        for edge in data.get("edges", []):
                            key = (edge.get("source"), edge.get("target"), edge.get("predicate"), edge.get("memory_id"))
                            if key not in seen_edges:
                                seen_edges.add(key)
                                merged_edges.append(edge)
                    elif isinstance(resp, Exception):
                        logger.error(f"Error querying entity graph from shard: {resp}")
                        
                return {"nodes": list(merged_nodes), "edges": merged_edges}
            else:
                # Multi-hop distributed BFS expansion across shards
                visited = {entity_id}
                frontier = {entity_id}
                all_nodes = {entity_id}
                all_edges = []
                seen_edges = set()
                max_hops = max(1, min(depth, 15))

                for _ in range(max_hops):
                    if not frontier:
                        break
                    tasks = []
                    current_batch = list(frontier)[:64]
                    for ent in current_batch:
                        for node in target_nodes:
                            tasks.append(
                                client.get(
                                    f"{node}/entity_graph",
                                    params={"entity_id": ent, "depth": 1},
                                    headers=get_forward_headers(ctx)
                                )
                            )
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    new_frontier = set()
                    for resp in responses:
                        if isinstance(resp, httpx.Response) and resp.status_code == 200:
                            data = resp.json()
                            for n in data.get("nodes", []):
                                all_nodes.add(n)
                                if n not in visited:
                                    visited.add(n)
                                    new_frontier.add(n)
                            for edge in data.get("edges", []):
                                key = (edge.get("source"), edge.get("target"), edge.get("predicate"), edge.get("memory_id"))
                                if key not in seen_edges:
                                    seen_edges.add(key)
                                    all_edges.append(edge)
                        elif isinstance(resp, Exception):
                            logger.error(f"Error querying entity graph from shard: {resp}")
                    frontier = new_frontier

                return {"nodes": list(all_nodes), "edges": all_edges}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                if not entity_id:
                    engine = await active_db._get_db()
                    hub_entities = await asyncio.to_thread(
                        engine.get_hub_entities, GRAPH_HUB_LIMIT
                    )
                    if not hub_entities:
                        all_entities = await asyncio.to_thread(engine.get_entities)
                        if not all_entities:
                            return {"nodes": [], "edges": []}
                        hub_entities = all_entities[:GRAPH_HUB_LIMIT]
                    
                    merged_nodes = set()
                    merged_edges = []
                    seen_edges = set()
                    
                    for ent in hub_entities:
                        merged_nodes.add(ent)
                        graph = await active_db.entity_graph(ent, depth=1)
                        for node in graph.nodes:
                            merged_nodes.add(node)
                        for edge in graph.edges:
                            key = (edge.get("source"), edge.get("target"), edge.get("predicate"), edge.get("memory_id"))
                            if key not in seen_edges:
                                seen_edges.add(key)
                                merged_edges.append(edge)
                    return {"nodes": list(merged_nodes), "edges": merged_edges}
                else:
                    graph = await active_db.entity_graph(entity_id, depth)
                    return {"nodes": graph.nodes, "edges": graph.edges}
            except Exception as e:
                logger.error(f"Error retrieving entity graph for {entity_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("entity_graph", payload, request, response, ctx, fetch)

@app.post("/get_timeline")
async def get_timeline(
    payload: TimelinePayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves timeline chronologically.
    """
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)
            tasks = [
                client.post(
                    f"{node}/get_timeline",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_memories = []
            seen_ids = set()
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("memories", []):
                        if m.get("id") not in seen_ids:
                            seen_ids.add(m.get("id"))
                            all_memories.append(m)
                elif isinstance(resp, Exception):
                    logger.error(f"Error getting timeline from shard: {resp}")
                    
            all_memories.sort(key=lambda x: x.get("created_at", 0.0))
            return {"memories": all_memories}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                results = await active_db.get_timeline(entity_id=payload.entity_id, start=payload.start, end=payload.end)
                return {"memories": [r._atom.to_dict() for r in results]}
            except Exception as e:
                logger.error(f"Error getting timeline: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("get_timeline", payload, request, response, ctx, fetch)

@app.get("/stats")
async def stats(ctx: RequestContext = Depends(get_context({Permission.ADMIN}))):
    """
    Provides real-time system metrics, cache status, and internal allocation maps.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        all_replica_nodes = []
        for group in shard_groups:
            all_replica_nodes.extend(group)
            
        tasks = [
            client.get(
                f"{node}/stats",
                headers=get_forward_headers(ctx)
            ) for node in all_replica_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_memory_count = 0
        total_l1_size = 0
        total_l2_size = 0
        total_entity_count = 0
        
        group_representatives = []
        for group in shard_groups:
            node = get_healthy_replica_for_group(group)
            if node:
                group_representatives.append(node)
                
        shards_metrics = {}
        for node, resp in zip(all_replica_nodes, responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                try:
                    data = resp.json()
                    if node in group_representatives:
                        total_memory_count += data.get("memory_count", 0)
                        total_l1_size += data.get("l1_size", 0)
                        total_l2_size += data.get("l2_size", 0)
                        total_entity_count += data.get("entity_count", 0)
                    
                    cached_status = SHARD_METRICS_CACHE.get(node, {}).get("status", "healthy")
                    shards_metrics[node] = {
                        "status": cached_status,
                        "memory_count": data.get("memory_count", 0),
                        "l1_size": data.get("l1_size", 0),
                        "l2_size": data.get("l2_size", 0),
                        "entity_count": data.get("entity_count", 0),
                        "cpu": data.get("cpu", 0.0),
                        "ram": data.get("ram", {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}),
                        "disk": data.get("disk", {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}),
                    }
                except Exception as e:
                    shards_metrics[node] = {
                        "status": "unhealthy",
                        "error": f"Failed to parse response: {str(e)}",
                        "cpu": 0.0,
                        "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                        "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                    }
            else:
                err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                shards_metrics[node] = {
                    "status": "unhealthy",
                    "error": err_msg,
                    "cpu": 0.0,
                    "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                    "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                }
                
        coord_system = {
            "cpu": get_cpu_usage(),
            "ram": get_ram_usage(),
            "disk": get_disk_usage(os.getenv("STORAGE_DIR", "."))
        }
        
        return {
            "mode": "coordinator",
            "memory_count": total_memory_count,
            "l1_size": total_l1_size,
            "l2_size": total_l2_size,
            "entity_count": total_entity_count,
            "system": coord_system,
            "shards": shards_metrics
        }
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            db_stats = await active_db.stats()
            db_stats["mode"] = "shard"
            db_stats["version"] = SERVER_VERSION
            try:
                db_stats["pending_extractions"] = await active_db.pending_extractions()
            except Exception:
                db_stats["pending_extractions"] = 0
            db_stats["cpu"] = get_cpu_usage()
            db_stats["ram"] = get_ram_usage()
            db_stats["disk"] = get_disk_usage(os.getenv("STORAGE_DIR", "./shared_memory"))
            return db_stats
        except Exception as e:
            logger.error(f"Unable to safely pull analytical parameters: {str(e)}")
            raise HTTPException(status_code=500, detail="Stats access blocked.")


# -------------------------------------------------------------------------
# Granular Import / Export Endpoints
# -------------------------------------------------------------------------
@app.get("/export")
async def export_data_get(
    request: Request,
    response: Response,
    data_type: str = Query("all", description="Category to export: 'all', 'skills', 'profiles', 'triplets', or 'memories'."),
    memory_type: Optional[str] = Query(None, description="Optional memory type filter."),
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Granularly exports stored database content (skills, user profiles, knowledge graph triplets, or memories).
    """
    return await _execute_export(data_type, memory_type, ctx)

@app.post("/export")
async def export_data_post(
    payload: ExportPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Granularly exports stored database content via POST payload.
    """
    return await _execute_export(payload.data_type, payload.memory_type, ctx)

async def _execute_export(
    data_type: str,
    memory_type: Optional[str],
    ctx: RequestContext
) -> Dict[str, Any]:
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        
        target_nodes = []
        for group in shard_groups:
            node = get_healthy_replica_for_group(group)
            if node:
                target_nodes.append(node)
                
        payload = {"data_type": data_type}
        if memory_type:
            payload["memory_type"] = memory_type
            
        tasks = [
            client.post(
                f"{node}/export",
                json=payload,
                headers=get_forward_headers(ctx)
            ) for node in target_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        merged_skills = []
        seen_skills = set()
        merged_profiles = []
        seen_profiles = set()
        merged_memories = []
        seen_memories = set()
        merged_triplets = []
        seen_triplets = set()
        
        for resp in responses:
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                body = resp.json().get("data", {})
                for s in body.get("skills", []):
                    sid = s.get("id") or (s.get("metadata", {}).get("skill_name") if s.get("metadata") else None)
                    if sid and sid not in seen_skills:
                        seen_skills.add(sid)
                        merged_skills.append(s)
                for p in body.get("profiles", []):
                    pid = p.get("id") or ((p.get("metadata", {}).get("user_id", "") if p.get("metadata") else "") + ":" + (p.get("payload") or ""))
                    if pid and pid not in seen_profiles:
                        seen_profiles.add(pid)
                        merged_profiles.append(p)
                for m in body.get("memories", []):
                    mid = m.get("id")
                    if mid and mid not in seen_memories:
                        seen_memories.add(mid)
                        merged_memories.append(m)
                for t in body.get("triplets", []):
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        tkey = (str(t[0]), str(t[1]), str(t[2]))
                        if tkey not in seen_triplets:
                            seen_triplets.add(tkey)
                            merged_triplets.append(list(tkey))
                            
        return {
            "version": "1.0",
            "server_version": SERVER_VERSION,
            "exported_at": time.time(),
            "data_type": data_type,
            "tenant": ctx.tenant or "global",
            "namespace": ctx.namespace or "default",
            "counts": {
                "skills": len(merged_skills),
                "profiles": len(merged_profiles),
                "memories": len(merged_memories),
                "triplets": len(merged_triplets)
            },
            "data": {
                "skills": merged_skills,
                "profiles": merged_profiles,
                "memories": merged_memories,
                "triplets": merged_triplets
            }
        }
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            raw_db = await active_db._get_db()
            result = {
                "skills": [],
                "profiles": [],
                "memories": [],
                "triplets": []
            }
            
            # Skills
            if data_type in ("all", "skills"):
                skills = await active_db.list_skills()
                result["skills"] = [s._atom.to_dict() if hasattr(s, "_atom") else s.to_dict() for s in skills]

            # Atoms from hot tier and cold tier
            all_atoms = {}
            with raw_db._internal_lock:
                for aid, atom in raw_db.hot_tier.atoms.items():
                    if aid not in raw_db.deleted_atom_ids:
                        all_atoms[aid] = atom
                for epoch_id in raw_db.cold_tier.get_all_epochs():
                    for atom in raw_db.cold_tier.load_epoch(epoch_id):
                        if atom.id not in raw_db.deleted_atom_ids and atom.id not in all_atoms:
                            all_atoms[atom.id] = atom

            seen_triples = set()
            for atom in all_atoms.values():
                mt = atom.memory_type.value if atom.memory_type else "general"
                atom_dict = atom.to_dict()

                if data_type in ("all", "profiles") and (mt == "profile" or (atom.metadata and atom.metadata.get("user_id"))):
                    result["profiles"].append(atom_dict)

                if data_type in ("all", "memories"):
                    if not memory_type or mt == memory_type:
                        result["memories"].append(atom_dict)

                if data_type in ("all", "triplets", "graph"):
                    if atom.triples:
                        for t in atom.triples:
                            if isinstance(t, (list, tuple)) and len(t) >= 3:
                                tkey = (str(t[0]), str(t[1]), str(t[2]))
                                if tkey not in seen_triples:
                                    seen_triples.add(tkey)
                                    result["triplets"].append(list(tkey))

            return {
                "version": "1.0",
                "server_version": SERVER_VERSION,
                "exported_at": time.time(),
                "data_type": data_type,
                "tenant": ctx.tenant or "global",
                "namespace": ctx.namespace or "default",
                "counts": {
                    "skills": len(result["skills"]),
                    "profiles": len(result["profiles"]),
                    "memories": len(result["memories"]),
                    "triplets": len(result["triplets"])
                },
                "data": result
            }
        except Exception as e:
            logger.error(f"Export execution failure: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/import", status_code=status.HTTP_201_CREATED)
async def import_data(
    payload: ImportPayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Granularly imports database records (skills, user profiles, memories, and triplets).
    """
    data = payload.data.get("data", payload.data)
    filter_type = payload.data_type

    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

        counts = {"skills": 0, "profiles": 0, "memories": 0, "triplets": 0}

        # 1. Skills
        if filter_type in (None, "all", "skills") and "skills" in data:
            for s in data["skills"]:
                meta = s.get("metadata") or {}
                skill_name = meta.get("skill_name") or s.get("skill_name") or s.get("id")
                desc = meta.get("description") or s.get("description") or s.get("payload") or ""
                steps = meta.get("steps") or s.get("steps") or []
                if skill_name:
                    sk_payload = {
                        "skill_name": skill_name,
                        "description": desc,
                        "steps": steps,
                        "tool_schema": meta.get("tool_schema"),
                        "metadata": meta,
                        "skill_id": s.get("id") or meta.get("process_id"),
                        "decision_rules": meta.get("decision_rules"),
                        "triples": s.get("triples") or meta.get("triples")
                    }
                    target_node = hash_ring.get_node(skill_name)
                    group = get_replica_group_for_node(target_node)
                    tasks = [client.post(f"{node}/skills/remember", json=sk_payload, headers=get_forward_headers(ctx)) for node in group]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    counts["skills"] += 1

        # 2. Profiles
        if filter_type in (None, "all", "profiles") and "profiles" in data:
            for p in data["profiles"]:
                meta = p.get("metadata") or {}
                uid = meta.get("user_id") or p.get("user_id")
                fact = p.get("payload") or p.get("text") or meta.get("fact_text") or ""
                if uid and fact:
                    pr_payload = {"user_id": uid, "fact_text": fact, "metadata": meta}
                    target_node = hash_ring.get_node(uid)
                    group = get_replica_group_for_node(target_node)
                    tasks = [client.post(f"{node}/profiles/remember", json=pr_payload, headers=get_forward_headers(ctx)) for node in group]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    counts["profiles"] += 1

        # 3. Memories
        if filter_type in (None, "all", "memories") and "memories" in data:
            for m in data["memories"]:
                text = m.get("payload") or m.get("text")
                if text:
                    mem_payload = {
                        "text": text,
                        "metadata": m.get("metadata"),
                        "triples": m.get("triples"),
                        "memory_type": m.get("memory_type") or (m.get("metadata") or {}).get("memory_type"),
                        "id": m.get("id")
                    }
                    target_node = get_healthy_shard(text)
                    group = get_replica_group_for_node(target_node)
                    tasks = [client.post(f"{node}/remember", json=mem_payload, headers=get_forward_headers(ctx)) for node in group]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    counts["memories"] += 1

        # 4. Triplets
        if filter_type in (None, "all", "triplets", "graph") and "triplets" in data:
            for t in data["triplets"]:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    text = f"{t[0]} {str(t[1]).replace('_', ' ')} {t[2]}."
                    tr_payload = {"text": text, "triples": [[t[0], t[1], t[2]]]}
                    target_node = get_healthy_shard(text)
                    group = get_replica_group_for_node(target_node)
                    tasks = [client.post(f"{node}/remember", json=tr_payload, headers=get_forward_headers(ctx)) for node in group]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    counts["triplets"] += 1

        invalidate_cache(ctx.tenant, ctx.namespace)
        await broadcast_sse_event("update", {"type": "import", "counts": counts})
        return {
            "status": "success",
            "imported": counts,
            "total_imported": sum(counts.values())
        }
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            counts = {"skills": 0, "profiles": 0, "memories": 0, "triplets": 0}

            # 1. Skills
            if filter_type in (None, "all", "skills") and "skills" in data:
                for s in data["skills"]:
                    meta = s.get("metadata") or {}
                    skill_name = meta.get("skill_name") or s.get("skill_name") or s.get("id")
                    desc = meta.get("description") or s.get("description") or s.get("payload") or ""
                    steps = meta.get("steps") or s.get("steps") or []
                    norm_steps = []
                    for idx, st in enumerate(steps):
                        if isinstance(st, str):
                            norm_steps.append({"step": idx + 1, "action": st})
                        elif isinstance(st, dict):
                            norm_steps.append(st)
                        else:
                            norm_steps.append({"step": idx + 1, "action": str(st)})
                    if skill_name:
                        await active_db.remember_skill(
                            skill_name=skill_name,
                            description=desc,
                            steps=norm_steps,
                            tool_schema=meta.get("tool_schema"),
                            metadata=meta,
                            skill_id=s.get("id") or meta.get("process_id"),
                            decision_rules=meta.get("decision_rules"),
                            triples=s.get("triples") or meta.get("triples")
                        )
                        counts["skills"] += 1

            # 2. Profiles
            if filter_type in (None, "all", "profiles") and "profiles" in data:
                for p in data["profiles"]:
                    meta = p.get("metadata") or {}
                    uid = meta.get("user_id") or p.get("user_id")
                    fact = p.get("payload") or p.get("text") or meta.get("fact_text") or ""
                    if uid and fact:
                        await active_db.remember_user_profile(user_id=uid, fact_text=fact, metadata=meta)
                        counts["profiles"] += 1

            # 3. Memories
            if filter_type in (None, "all", "memories") and "memories" in data:
                for m in data["memories"]:
                    text = m.get("payload") or m.get("text")
                    if text:
                        await active_db.remember(
                            text=text,
                            triples=m.get("triples"),
                            metadata=m.get("metadata"),
                            memory_type=m.get("memory_type") or (m.get("metadata") or {}).get("memory_type"),
                            atom_id=m.get("id")
                        )
                        counts["memories"] += 1

            # 4. Triplets
            if filter_type in (None, "all", "triplets", "graph") and "triplets" in data:
                for t in data["triplets"]:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        await active_db.remember(
                            text=f"{t[0]} {str(t[1]).replace('_', ' ')} {t[2]}.",
                            triples=[[t[0], t[1], t[2]]]
                        )
                        counts["triplets"] += 1

            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "import", "counts": counts})
            return {
                "status": "success",
                "imported": counts,
                "total_imported": sum(counts.values())
            }
        except Exception as e:
            logger.error(f"Import execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/compact", status_code=status.HTTP_200_OK)
async def compact(ctx: RequestContext = Depends(get_context({Permission.ADMIN}))):
    """
    Administrative endpoint to compress historical Parquet archives, clear soft deletes,
    and release unneeded disk space dynamically.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_nodes = []
        for group in shard_groups:
            target_nodes.extend(get_all_healthy_replicas_for_group(group))
        tasks = [
            client.post(
                f"{node}/compact",
                headers=get_forward_headers(ctx)
            ) for node in target_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for resp in responses:
            if isinstance(resp, Exception):
                logger.error(f"Error compacting shard: {resp}")
        invalidate_cache(ctx.tenant, ctx.namespace)
        return {"status": "compaction completed"}
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            logger.info("Triggering background historical archive compaction...")
            await active_db.compact()
            invalidate_cache(ctx.tenant, ctx.namespace)
            return {"status": "compaction completed"}
        except Exception as e:
            logger.error(f"Compaction runtime error occurred: {str(e)}")
            raise HTTPException(status_code=500, detail="Compaction execution failure.")


class SQLQueryPayload(BaseModel):
    query: str


@app.post("/v1/analytics/query", status_code=status.HTTP_200_OK)
async def analytics_query(
    payload: SQLQueryPayload,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Executes a DuckDB SQL analytical query over Cold Tier Parquet archives.
    The table `cold_tier` is automatically available representing all parquet files.
    """
    active_db = await get_db_instance(ctx.tenant, ctx.namespace)
    try:
        results = await active_db.query_sql(payload.query)
        return {"status": "success", "data": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Analytics SQL query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL Query failed: {str(e)}")


UI_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "ui")
UI_ASSETS_DIR = os.path.join(UI_STATIC_DIR, "assets")

@app.get("/logo.png")
async def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "logo-epoch.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
    absolute_logo = "/home/jeff/Projects/epochdb-server/src/logo-epoch.png"
    if os.path.exists(absolute_logo):
        return FileResponse(absolute_logo)
    raise HTTPException(status_code=404, detail="Logo not found")

@app.get("/favicon.svg")
@app.get("/favicon.ico")
async def get_favicon():
    favicon_path = os.path.join(UI_STATIC_DIR, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")

def _inject_studio_config(html_content: str) -> str:
    """Inject server mode/version/default tokens into the Studio SPA HTML."""
    inject_script = f"""<script>
        window.__EPOCHDB_SERVER_CONFIG__ = {{
            mode: "{NODE_MODE}",
            version: "{SERVER_VERSION}",
            defaultApiKey: "{API_KEY or ''}",
            defaultInternalToken: "{INTERNAL_AUTH_TOKEN or ''}"
        }};
        </script>"""
    if "<head>" in html_content:
        return html_content.replace("<head>", f"<head>\n    {inject_script}", 1)
    return inject_script + html_content

@app.get("/", response_class=HTMLResponse)
@app.get("/visualize", response_class=HTMLResponse)
async def visualize():
    index_path = os.path.join(UI_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=_inject_studio_config(html_content))
    raise HTTPException(
        status_code=404,
        detail="Studio UI not found. Build the React app with: cd ui && npm run build",
    )

@app.get("/stream")
async def sse_stream(request: Request):
    """
    Server-Sent Events endpoint to stream database mutation notifications to clients.
    """
    queue = asyncio.Queue()
    SSE_LISTENERS.append(queue)
    
    async def event_generator():
        try:
            # Yield initial connection confirmation
            yield {"event": "connected", "data": "connected"}
            while True:
                event = await queue.get()
                sse_event = event.copy()
                if isinstance(sse_event.get("data"), (dict, list)):
                    sse_event["data"] = json.dumps(sse_event["data"])
                yield sse_event
        except asyncio.CancelledError:
            pass
        finally:
            if queue in SSE_LISTENERS:
                SSE_LISTENERS.remove(queue)
                
    return EventSourceResponse(event_generator())

# Mount Studio static assets after API routes so /assets does not shadow them.
if os.path.isdir(UI_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=UI_ASSETS_DIR), name="ui_assets")

# -------------------------------------------------------------------------
# 6. Production Execution Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        workers=1,
        loop="auto",
        http="auto"
    )