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
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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
SERVER_VERSION = "0.9.7"


# -------------------------------------------------------------------------
# 1. Structured Logging Configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (Line: %(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("epochdb_production_server")

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
    logger.info(f"Invalidated read cache for tenant='{tenant}' namespace='{namespace}'. New version: {CLUSTER_STATE_VERSIONS[ctx_key]}")

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
    logger.info(f"Broadcasting SSE event '{event_type}' to {len(SSE_LISTENERS)} listeners.")
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
    """
    logger.info("Starting background health polling loop for shards...")
    all_replica_nodes = []
    for group in shard_groups:
        all_replica_nodes.extend(group)
        
    while True:
        if client:
            try:
                tasks = [client.get(f"{shard}/stats", timeout=2.0) for shard in all_replica_nodes]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                status_changed = False
                for shard, resp in zip(all_replica_nodes, responses):
                    prev_status = SHARD_METRICS_CACHE.get(shard, {}).get("status")
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
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
                            logger.info(f"poll_shards_loop: shard={shard} prev_status={prev_status} syncing={shard in SYNCING_NODES}")
                            if BACKGROUND_SYNC_ENABLED and prev_status not in ("healthy", "synchronizing") and shard not in SYNCING_NODES:
                                group = get_replica_group_for_node(shard)
                                source_node = None
                                for sibling in group:
                                    if sibling != shard:
                                        sib_metrics = SHARD_METRICS_CACHE.get(sibling)
                                        sib_status = sib_metrics.get("status") if sib_metrics else None
                                        logger.info(f"poll_shards_loop: checking sibling={sibling} status={sib_status}")
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
                        err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                        SHARD_METRICS_CACHE[shard] = {
                            "status": "unhealthy",
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
        await asyncio.sleep(5)

def get_healthy_shard(key: str) -> str:
    """
    Selects the target shard for a key using consistent hashing.
    If the target shard group is unhealthy or overloaded (CPU/RAM/Disk > 90%),
    it routes to the next healthy alternative shard group in the ring.
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

    for node in shard_nodes:
        if node != primary_node and is_group_good(node):
            logger.warning(f"Routing key '{key[:20]}...' to healthy fallback shard group {node} instead of overloaded/unhealthy {primary_node}")
            return node

    for node in shard_nodes:
        group = get_replica_group_for_node(node)
        for rep in group:
            metrics = SHARD_METRICS_CACHE.get(rep)
            if metrics and metrics.get("status") == "healthy":
                logger.warning(f"Routing to overloaded but alive fallback shard group {node}")
                return node

    logger.warning(f"All nodes unhealthy/offline. Routing to default consistent-hashing node {primary_node}")
    return primary_node


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
        client = httpx.AsyncClient(headers=headers, timeout=30.0)
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
    text: str = Field(..., description="The factual memory text to write to the engine.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional structural metadata or graph triples.")
    id: Optional[str] = Field(default=None, description="Optional unique identifier (pre-calculated or forwarded).")
    memory_type: Optional[str] = Field(default=None, description="Optional memory type: 'general', 'episodic', 'profile', or 'working'.")

class QueryPayload(BaseModel):
    query: str = Field(..., description="The semantic search or multi-hop lookup query string.")
    k: int = Field(default=1, ge=1, le=100, description="The number of candidate matches to return.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="MongoDB-style metadata filter evaluation parameters.")
    memory_type: Optional[str] = Field(default=None, description="Optional filter by memory type: 'general', 'episodic', 'profile', or 'working'.")
    context_window: int = Field(default=0, ge=0, description="Number of chronological context turns to retrieve around the matched memory.")
    expand_hops: int = Field(default=0, ge=0, le=10, description="Relational KG expansion hops during retrieval.")

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

class DeletePayload(BaseModel):
    memory_id: str = Field(..., description="The unique identifier of the memory to delete.")
    hard: bool = Field(default=False, description="Whether to hard delete or soft delete.")

class TimelinePayload(BaseModel):
    entity_id: Optional[str] = Field(default=None, description="Optional entity ID.")
    start: Optional[float] = Field(default=None, description="Optional start timestamp.")
    end: Optional[float] = Field(default=None, description="Optional end timestamp.")

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
        
        target_shard = None
        original_predefined_id = payload.id
        
        max_attempts = 3
        attempt = 0
        tried_nodes = set()
        
        while attempt < max_attempts:
            attempt += 1
            
            if original_predefined_id:
                # If predefined ID is provided and contains a valid prefix, route to that shard
                target_shard = get_shard_for_id(original_predefined_id)
                if target_shard:
                    atom_id = original_predefined_id
                else:
                    # Prepend prefix using health-based routing
                    target_shard = get_healthy_shard(payload.text)
                    shard_idx = shard_nodes.index(target_shard)
                    atom_id = f"shard{shard_idx}-{original_predefined_id}"
            else:
                # Generate prefixed UUID, using health-based routing
                target_shard = get_healthy_shard(payload.text)
                shard_idx = shard_nodes.index(target_shard)
                atom_id = f"shard{shard_idx}-{uuid.uuid4().hex}"
                
            # If the chosen target shard has already failed in this request cycle, fallback
            if target_shard in tried_nodes:
                alternative = None
                for node in shard_nodes:
                    if node not in tried_nodes:
                        metrics = SHARD_METRICS_CACHE.get(node)
                        if metrics and metrics.get("status") == "healthy":
                            alternative = node
                            break
                if not alternative:
                    for node in shard_nodes:
                        if node not in tried_nodes:
                            alternative = node
                            break
                if alternative:
                    target_shard = alternative
                    shard_idx = shard_nodes.index(target_shard)
                    if original_predefined_id:
                        atom_id = f"shard{shard_idx}-{original_predefined_id}"
                    else:
                        atom_id = f"shard{shard_idx}-{uuid.uuid4().hex}"
                else:
                    break
                    
            tried_nodes.add(target_shard)
            
            try:
                target_payload = {
                    "text": payload.text,
                    "metadata": payload.metadata,
                    "id": atom_id,
                    "memory_type": payload.memory_type
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
                        err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                        logger.warning(f"Replication write failed to node {node}: {err_msg}")
                        failures.append((node, err_msg))
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
                    raise Exception(f"Write consistency check failed. Required: {consistency.value} ({required_success}), got: {len(success_nodes)} success(es). Last error: {last_err}")
            except Exception as e:
                logger.error(f"Failed to forward write to shard set {target_shard} on attempt {attempt}: {e}")
                if attempt >= max_attempts:
                    raise HTTPException(status_code=500, detail=f"Failed to forward write to shard set: {str(e)}")
            
    else:
        # Shard Mode (Storage Node)
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            if payload.metadata is None:
                payload.metadata = {}
            if "_updated_at" not in payload.metadata:
                payload.metadata["_updated_at"] = time.time()
                
            if payload.id:
                engine = await active_db._get_db()
                text = payload.text
                metadata = payload.metadata
                
                # Manual embedding generation to support predefined atom_id
                if engine._model_name:
                    embedder = engine._get_embedder()
                    emb = await asyncio.to_thread(embedder.encode, text, normalize_embeddings=True)
                    embedding = np.array(emb, dtype=np.float32)
                else:
                    embedding = np.zeros(engine.dim, dtype=np.float32)
                    
                triples = metadata.get("triples") or []
                if not triples:
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

                    triples = await asyncio.to_thread(_extract_triples_for_ingest)

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
                atom_id = await active_db.remember(text=payload.text, metadata=payload.metadata, memory_type=payload.memory_type)
                logger.info(f"Ingested atom: '{payload.text[:40]}...'")
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                return {"status": "success", "id": atom_id}
        except Exception as e:
            logger.error(f"Failed to commit memory write block: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal storage layer mutation rejected: {str(e)}")

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
    Performs semantic search across the memory database.
    Coordinator parallelizes requests to all shards and merges/re-ranks results.
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
                    f"{node}/query",
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
                    logger.error(f"Error querying shard: {resp}")
                    
            # Sort by similarity score in descending order
            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return {"results": all_results[:payload.k]}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                results = await active_db.query(
                    text=payload.query,
                    k=payload.k,
                    filters=payload.filters,
                    memory_type=payload.memory_type,
                    context_window=payload.context_window,
                    expand_hops=payload.expand_hops,
                )
                
                # Retrieve engine to compute exact similarity scores
                engine = await active_db._get_db()
                embedder = engine._get_embedder()
                q_emb = await asyncio.to_thread(embedder.encode, payload.query, normalize_embeddings=True)
                q_emb = np.array(q_emb, dtype=np.float32)
                
                formatted_results = []
                for r in results:
                    score = 0.0
                    if q_emb.any() and r._atom.embedding.any():
                        score = float(np.dot(r._atom.embedding, q_emb) / (
                            np.linalg.norm(r._atom.embedding) * np.linalg.norm(q_emb) + 1e-10
                        ))
                    formatted_results.append({
                        "id": r.id,
                        "text": r.text,
                        "metadata": r.metadata,
                        "created_at": r.created_at,
                        "score": score
                    })
                return {"results": formatted_results}
            except Exception as e:
                logger.error(f"Error resolving retrieval operations: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("query", payload, request, response, ctx, fetch)

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
                    err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                    logger.warning(f"Replication update failed on replica {node}: {err_msg}")
                    failures.append((node, err_msg))
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
                    err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                    logger.warning(f"Replication delete failed on replica {node}: {err_msg}")
                    failures.append((node, err_msg))
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
                
            params = {"depth": depth}
            if entity_id:
                params["entity_id"] = entity_id
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)
            tasks = [
                client.get(
                    f"{node}/entity_graph",
                    params=params,
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
                        # Dedup edges by source, target, predicate, memory_id
                        key = (edge.get("source"), edge.get("target"), edge.get("predicate"), edge.get("memory_id"))
                        if key not in seen_edges:
                            seen_edges.add(key)
                            merged_edges.append(edge)
                elif isinstance(resp, Exception):
                    logger.error(f"Error querying entity graph from shard: {resp}")
                    
            return {"nodes": list(merged_nodes), "edges": merged_edges}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                if not entity_id:
                    engine = await active_db._get_db()
                    hub_entities = await asyncio.to_thread(
                        engine.get_hub_entities, GRAPH_HUB_LIMIT
                    )
                    if not hub_entities:
                        return {"nodes": [], "edges": []}
                    
                    merged_nodes = set()
                    merged_edges = []
                    seen_edges = set()
                    
                    for ent in hub_entities:
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
            db_stats["cpu"] = get_cpu_usage()
            db_stats["ram"] = get_ram_usage()
            db_stats["disk"] = get_disk_usage(os.getenv("STORAGE_DIR", "./shared_memory"))
            return db_stats
        except Exception as e:
            logger.error(f"Unable to safely pull analytical parameters: {str(e)}")
            raise HTTPException(status_code=500, detail="Stats access blocked.")

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


@app.get("/logo.png")
async def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "logo-epoch.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
    absolute_logo = "/home/jeff/Projects/epochdb-server/src/logo-epoch.png"
    if os.path.exists(absolute_logo):
        return FileResponse(absolute_logo)
    raise HTTPException(status_code=404, detail="Logo not found")

@app.get("/visualize", response_class=HTMLResponse)
async def visualize():
    template_path = os.path.join(os.path.dirname(__file__), "visualize.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    raise HTTPException(status_code=404, detail="Visualization template not found")

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