# src/server.py
import logging
import os
import uvicorn
import uuid
import hashlib
import bisect
import asyncio
import httpx
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from epochdb import AsyncEpochDB


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
shard_nodes = [s.strip() for s in shard_nodes_str.split(",") if s.strip()]
hash_ring = ConsistentHashRing(shard_nodes) if shard_nodes else None

# Security credentials and API key headers configuration
API_KEY = os.getenv("API_KEY")
INTERNAL_AUTH_TOKEN = os.getenv("INTERNAL_AUTH_TOKEN")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
internal_token_header = APIKeyHeader(name="X-Internal-Token", auto_error=False)

async def verify_auth(
    x_api_key: Optional[str] = Security(api_key_header),
    x_internal_token: Optional[str] = Security(internal_token_header)
):
    if NODE_MODE == "coordinator":
        if API_KEY and x_api_key != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header."
            )
    else:
        token = x_internal_token or x_api_key
        if INTERNAL_AUTH_TOKEN and token != INTERNAL_AUTH_TOKEN:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-Internal-Token header."
            )


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
# Global reference for the async database engine instance (Shard Mode)
db: Optional[AsyncEpochDB] = None
# Global HTTP client session (Coordinator Mode)
client: Optional[httpx.AsyncClient] = None

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
        yield
        await client.aclose()
        logger.info("Coordinator HTTP client session closed cleanly.")
    else:
        logger.info("Initializing high-performance AsyncEpochDB engine in Shard Mode...")
        storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
        try:
            async with AsyncEpochDB(
                storage_dir=storage_dir,
                embedding_model="all-MiniLM-L6-v2",
                wal_sync_interval=0.1,
                parquet_compression="zstd",
                parquet_compression_level=3
            ) as engine:
                db = engine
                
                # --- WARM-UP SEQUENCE ---
                logger.info("Warming up embedding model (this may take a few seconds)...")
                await db.query(text="system boot warmup", k=1)
                logger.info("Model warmed up. AsyncEpochDB engine successfully mounted and listening.")
                
                yield  # Server begins accepting HTTP requests here
                
        except Exception as e:
            logger.critical(f"Fatal error during engine startup sequence: {str(e)}")
            raise e
        finally:
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

class QueryPayload(BaseModel):
    query: str = Field(..., description="The semantic search or multi-hop lookup query string.")
    k: int = Field(default=1, ge=1, le=100, description="The number of candidate matches to return.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="MongoDB-style metadata filter evaluation parameters.")

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
        return {"status": "healthy", "mode": "coordinator"}
    else:
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        return {"status": "healthy", "mode": "shard"}

@app.post("/remember", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_auth)])
async def remember(payload: MemoryPayload):
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
        atom_id = payload.id
        
        if atom_id:
            # If predefined ID is provided and contains a valid prefix, route to that shard
            target_shard = get_shard_for_id(atom_id)
            if not target_shard:
                # Prepend prefix if it's not prefixed
                target_shard = hash_ring.get_node(payload.text)
                shard_idx = shard_nodes.index(target_shard)
                atom_id = f"shard{shard_idx}-{atom_id}"
        else:
            # Generate prefixed UUID
            target_shard = hash_ring.get_node(payload.text)
            shard_idx = shard_nodes.index(target_shard)
            atom_id = f"shard{shard_idx}-{uuid.uuid4().hex}"
            
        try:
            target_payload = {
                "text": payload.text,
                "metadata": payload.metadata,
                "id": atom_id
            }
            res = await client.post(f"{target_shard}/remember", json=target_payload)
            res.raise_for_status()
            return {"status": "success", "id": atom_id}
        except Exception as e:
            logger.error(f"Failed to forward write to shard {target_shard}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to forward write to shard: {str(e)}")
            
    else:
        # Shard Mode (Storage Node)
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            if payload.id:
                engine = await db._get_db()
                text = payload.text
                metadata = payload.metadata or {}
                
                # Manual embedding generation to support predefined atom_id
                if engine._model_name:
                    embedder = engine._get_embedder()
                    emb = await asyncio.to_thread(embedder.encode, text, normalize_embeddings=True)
                    embedding = np.array(emb, dtype=np.float32)
                else:
                    embedding = np.zeros(engine.dim, dtype=np.float32)
                    
                triples = metadata.get("triples") or []
                if not triples:
                    extracted = await asyncio.to_thread(engine.extract_entities, text)
                    triples = [(str(e), "mentions", str(e)) for e in extracted]
                    
                atom_id = await asyncio.to_thread(
                    engine.add_memory,
                    payload=text,
                    embedding=embedding,
                    triples=triples,
                    metadata=metadata,
                    atom_id=payload.id
                )
                logger.info(f"Ingested atom with fixed ID {atom_id}: '{text[:40]}...'")
                return {"status": "success", "id": atom_id}
            else:
                atom_id = await db.remember(text=payload.text, metadata=payload.metadata)
                logger.info(f"Ingested atom: '{payload.text[:40]}...'")
                return {"status": "success", "id": atom_id}
        except Exception as e:
            logger.error(f"Failed to commit memory write block: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal storage layer mutation rejected: {str(e)}")

@app.post("/get", dependencies=[Depends(verify_auth)])
async def get_memory(payload: GetPayload):
    """
    Retrieves a specific memory by its unique ID.
    Coordinator routes directly if ID is prefixed, otherwise broadcasts.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_shard = get_shard_for_id(payload.memory_id)
        if target_shard:
            try:
                res = await client.post(f"{target_shard}/get", json=payload.model_dump())
                res.raise_for_status()
                return res.json()
            except Exception as e:
                logger.error(f"Error forwarding get to shard {target_shard}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            # Broadcast to all shards in parallel
            tasks = [client.post(f"{shard}/get", json=payload.model_dump()) for shard in shard_nodes]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    if data and "id" in data:
                        return data
            return {}
    else:
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            mem = await db.get(payload.memory_id)
            if mem:
                return mem._atom.to_dict()
            return {}
        except Exception as e:
            logger.error(f"Error resolving memory retrieval: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", dependencies=[Depends(verify_auth)])
async def query_memories(payload: QueryPayload):
    """
    Performs semantic search across the memory database.
    Coordinator parallelizes requests to all shards and merges/re-ranks results.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        tasks = [client.post(f"{shard}/query", json=payload.model_dump()) for shard in shard_nodes]
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
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            results = await db.query(text=payload.query, k=payload.k, filters=payload.filters)
            
            # Retrieve engine to compute exact similarity scores
            engine = await db._get_db()
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

@app.post("/update", dependencies=[Depends(verify_auth)])
async def update_memory(payload: UpdatePayload):
    """
    Updates memory text or metadata.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_shard = get_shard_for_id(payload.memory_id)
        if target_shard:
            try:
                res = await client.post(f"{target_shard}/update", json=payload.model_dump())
                res.raise_for_status()
                return res.json()
            except Exception as e:
                logger.error(f"Error forwarding update to shard {target_shard}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            # Broadcast to all
            tasks = [client.post(f"{shard}/update", json=payload.model_dump()) for shard in shard_nodes]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for resp in responses:
                if isinstance(resp, Exception):
                    logger.error(f"Error updating shard: {resp}")
            return {"status": "success"}
    else:
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            await db.update(payload.memory_id, payload.text, payload.metadata)
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Error updating memory {payload.memory_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete", dependencies=[Depends(verify_auth)])
async def delete_memory(payload: DeletePayload):
    """
    Deletes memory (hard or soft).
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_shard = get_shard_for_id(payload.memory_id)
        if target_shard:
            try:
                res = await client.post(f"{target_shard}/delete", json=payload.model_dump())
                res.raise_for_status()
                return res.json()
            except Exception as e:
                logger.error(f"Error forwarding delete to shard {target_shard}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            # Broadcast
            tasks = [client.post(f"{shard}/delete", json=payload.model_dump()) for shard in shard_nodes]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for resp in responses:
                if isinstance(resp, Exception):
                    logger.error(f"Error deleting shard: {resp}")
            return {"status": "success"}
    else:
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            await db.delete(payload.memory_id, payload.hard)
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Error deleting memory {payload.memory_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/entity_graph", dependencies=[Depends(verify_auth)])
async def entity_graph(entity_id: str, depth: int = 2):
    """
    Retrieves the local entity graph or aggregates the distributed graph.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        tasks = [client.get(f"{shard}/entity_graph?entity_id={entity_id}&depth={depth}") for shard in shard_nodes]
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
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            graph = await db.entity_graph(entity_id, depth)
            return {"nodes": graph.nodes, "edges": graph.edges}
        except Exception as e:
            logger.error(f"Error retrieving entity graph for {entity_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_timeline", dependencies=[Depends(verify_auth)])
async def get_timeline(payload: TimelinePayload):
    """
    Retrieves timeline chronologically.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        tasks = [client.post(f"{shard}/get_timeline", json=payload.model_dump()) for shard in shard_nodes]
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
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            results = await db.get_timeline(entity_id=payload.entity_id, start=payload.start, end=payload.end)
            return {"memories": [r._atom.to_dict() for r in results]}
        except Exception as e:
            logger.error(f"Error getting timeline: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", status_code=status.HTTP_200_OK, dependencies=[Depends(verify_auth)])
async def stats():
    """
    Provides real-time system metrics, cache status, and internal allocation maps.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        tasks = [client.get(f"{shard}/stats") for shard in shard_nodes]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_memory_count = 0
        total_l1_size = 0
        total_l2_size = 0
        total_entity_count = 0
        
        for resp in responses:
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                data = resp.json()
                total_memory_count += data.get("memory_count", 0)
                total_l1_size += data.get("l1_size", 0)
                total_l2_size += data.get("l2_size", 0)
                total_entity_count += data.get("entity_count", 0)
                
        return {
            "memory_count": total_memory_count,
            "l1_size": total_l1_size,
            "l2_size": total_l2_size,
            "entity_count": total_entity_count
        }
    else:
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            return await db.stats()
        except Exception as e:
            logger.error(f"Unable to safely pull analytical parameters: {str(e)}")
            raise HTTPException(status_code=500, detail="Stats access blocked.")

@app.post("/compact", status_code=status.HTTP_200_OK, dependencies=[Depends(verify_auth)])
async def compact():
    """
    Administrative endpoint to compress historical Parquet archives, clear soft deletes,
    and release unneeded disk space dynamically.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        tasks = [client.post(f"{shard}/compact") for shard in shard_nodes]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for resp in responses:
            if isinstance(resp, Exception):
                logger.error(f"Error compacting shard: {resp}")
        return {"status": "compaction completed"}
    else:
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        try:
            logger.info("Triggering background historical archive compaction...")
            await db.compact()
            return {"status": "compaction completed"}
        except Exception as e:
            logger.error(f"Compaction runtime error occurred: {str(e)}")
            raise HTTPException(status_code=500, detail="Compaction execution failure.")

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