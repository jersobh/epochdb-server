import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

import sys
# Configure environment variables BEFORE importing app so they are read during module load
temp_dir = tempfile.mkdtemp(prefix="epochdb_server_api_test_")
os.environ["NODE_MODE"] = "shard"
os.environ["INTERNAL_AUTH_TOKEN"] = "test-token-12345"
os.environ["STORAGE_DIR"] = temp_dir

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.server import app

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    shutil.rmtree(temp_dir, ignore_errors=True)

class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
        
    def json(self):
        return self._json_data
        
    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("Error", request=None, response=None)

class MockAsyncClient:
    async def post(self, url, **kwargs):
        if "/remember" in url:
            return MockResponse(201, {"status": "success", "id": kwargs.get("json", {}).get("id", "test-id")})
        elif "/get" in url:
            return MockResponse(200, {"id": "test-id", "payload": "Water boils at 100 degrees Celsius."})
        elif "/query" in url:
            return MockResponse(200, {"results": [{"id": "test-id", "score": 0.95}]})
        elif "/update" in url or "/delete" in url:
            return MockResponse(200, {"status": "success"})
        elif "/compact" in url:
            return MockResponse(200, {"status": "compaction completed"})
        elif "/get_timeline" in url:
            return MockResponse(200, {"memories": []})
        return MockResponse(404, {})
        
    async def get(self, url, **kwargs):
        if "/healthz" in url:
            return MockResponse(200, {"status": "healthy", "mode": "shard"})
        elif "/entity_graph" in url:
            return MockResponse(200, {"nodes": ["Apple"], "edges": []})
        elif "/stats" in url:
            return MockResponse(200, {"memory_count": 5, "l1_size": 2})
        return MockResponse(404, {})

    async def aclose(self):
        pass

def test_unauthorized_access():
    """Verify that requests with invalid or missing auth tokens return 401 Unauthorized."""
    with TestClient(app) as client:
        # Missing token
        resp = client.post("/remember", json={"text": "Factual memory text"})
        assert resp.status_code == 401
        
        # Wrong token
        resp = client.post("/remember", json={"text": "Factual memory text"}, headers={"X-Internal-Token": "bad-token"})
        assert resp.status_code == 401

def test_healthz():
    """Verify the liveness/readiness check works properly."""
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy", "mode": "shard"}

def test_crud_lifecycle():
    """Verify standard CRUD lifecycle on shard node endpoints (remember, get, query, update, delete)."""
    headers = {"X-Internal-Token": "test-token-12345"}
    
    with TestClient(app) as client:
        # 1. /remember
        resp = client.post("/remember", json={"text": "Water boils at 100 degrees Celsius."}, headers=headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "success"
        mem_id = data["id"]
        assert isinstance(mem_id, str)
        
        # 2. /get
        resp = client.post("/get", json={"memory_id": mem_id}, headers=headers)
        assert resp.status_code == 200
        mem_data = resp.json()
        assert mem_data["id"] == mem_id
        assert mem_data["payload"] == "Water boils at 100 degrees Celsius."
        
        # 3. /query
        resp = client.post("/query", json={"query": "boiling water", "k": 1}, headers=headers)
        assert resp.status_code == 200
        query_data = resp.json()
        assert "results" in query_data
        assert len(query_data["results"]) == 1
        assert query_data["results"][0]["id"] == mem_id
        assert "score" in query_data["results"][0]
        
        # 4. /update
        resp = client.post("/update", json={"memory_id": mem_id, "text": "Water boils at 100 degrees Celsius at sea level."}, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"status": "success"}
        
        # Get updated memory
        resp = client.post("/get", json={"memory_id": mem_id}, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["payload"] == "Water boils at 100 degrees Celsius at sea level."
        
        # 5. /delete (soft delete)
        resp = client.post("/delete", json={"memory_id": mem_id, "hard": False}, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"status": "success"}
        
        # Get soft-deleted memory (should return empty)
        resp = client.post("/get", json={"memory_id": mem_id}, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {}

def test_entity_graph_and_timeline():
    """Verify entity graph building and timeline retrieval."""
    headers = {"X-Internal-Token": "test-token-12345"}
    
    with TestClient(app) as client:
        # Ingest memory with explicit entity triples
        resp = client.post(
            "/remember",
            json={
                "text": "Steve Jobs founded Apple.",
                "metadata": {"triples": [("Steve Jobs", "founded", "Apple")]}
            },
            headers=headers
        )
        assert resp.status_code == 201
        
        # Query entity graph with specific entity_id
        resp = client.get("/entity_graph?entity_id=Apple&depth=1", headers=headers)
        assert resp.status_code == 200
        graph_data = resp.json()
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "Apple" in graph_data["nodes"]
        
        # Query entity graph without specific entity_id (returns default/all entities)
        resp = client.get("/entity_graph", headers=headers)
        assert resp.status_code == 200
        graph_data = resp.json()
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "Apple" in graph_data["nodes"]
        
        # Query timeline
        resp = client.post("/get_timeline", json={"entity_id": "Steve Jobs"}, headers=headers)
        assert resp.status_code == 200
        timeline_data = resp.json()
        assert "memories" in timeline_data
        assert len(timeline_data["memories"]) >= 1

def test_stats_and_compaction():
    """Verify stats retrieval and compaction endpoints."""
    headers = {"X-Internal-Token": "test-token-12345"}
    
    with TestClient(app) as client:
        # Check stats
        resp = client.get("/stats", headers=headers)
        assert resp.status_code == 200
        stats = resp.json()
        assert "memory_count" in stats
        assert "l1_size" in stats
        assert "cpu" in stats
        assert "ram" in stats
        assert "disk" in stats
        
        # Trigger compact
        resp = client.post("/compact", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"status": "compaction completed"}

def test_hard_delete():
    """Verify hard delete functionality removes the record completely."""
    headers = {"X-Internal-Token": "test-token-12345"}
    with TestClient(app) as client:
        # Ingest memory
        resp = client.post("/remember", json={"text": "Temporary deletion test text."}, headers=headers)
        assert resp.status_code == 201
        mem_id = resp.json()["id"]
        
        # Hard delete
        resp = client.post("/delete", json={"memory_id": mem_id, "hard": True}, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"status": "success"}
        
        # Get should return empty
        resp = client.post("/get", json={"memory_id": mem_id}, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {}


def test_coordinator_mode():
    """Verify coordinator mode routing, consistent hashing, and broadcasting using mocked shard nodes."""
    # 1. Back up original globals
    import src.server as server
    orig_mode = server.NODE_MODE
    orig_shards = server.shard_nodes
    orig_client = server.client
    orig_hash_ring = server.hash_ring
    orig_api_key = server.API_KEY
    
    # 2. Configure coordinator state
    server.NODE_MODE = "coordinator"
    server.API_KEY = "coord-api-key"
    server.shard_nodes = ["http://shard0:8000", "http://shard1:8000"]
    from src.server import ConsistentHashRing
    server.hash_ring = ConsistentHashRing(server.shard_nodes)
    
    headers = {"X-API-Key": "coord-api-key"}
    
    try:
        with TestClient(app) as client:
            # Overwrite client AFTER startup lifespan has finished executing
            server.client = MockAsyncClient()
            
            # Test Healthz
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json() == {"status": "healthy", "mode": "coordinator"}
            
            # Test Remember
            resp = client.post("/remember", json={"text": "Hello shard world"}, headers=headers)
            assert resp.status_code == 201
            assert resp.json()["status"] == "success"
            
            # Test Get (routes directly because id prefix is provided)
            resp = client.post("/get", json={"memory_id": "shard0-abc"}, headers=headers)
            assert resp.status_code == 200
            
            # Test Get (broadcasts to all because id prefix is missing)
            resp = client.post("/get", json={"memory_id": "nonprefixed-id"}, headers=headers)
            assert resp.status_code == 200
            
            # Test Query
            resp = client.post("/query", json={"query": "shard query"}, headers=headers)
            assert resp.status_code == 200
            
            # Test Update (routes directly because id prefix is provided)
            resp = client.post("/update", json={"memory_id": "shard1-def", "text": "updated"}, headers=headers)
            assert resp.status_code == 200
            
            # Test Update (broadcasts because prefix is missing)
            resp = client.post("/update", json={"memory_id": "nonprefixed", "text": "updated"}, headers=headers)
            assert resp.status_code == 200
            
            # Test Delete
            resp = client.post("/delete", json={"memory_id": "shard0-ghi"}, headers=headers)
            assert resp.status_code == 200
            
            # Test Delete (broadcasts because prefix is missing)
            resp = client.post("/delete", json={"memory_id": "nonprefixed"}, headers=headers)
            assert resp.status_code == 200
            
            # Test Entity Graph
            resp = client.get("/entity_graph?entity_id=Apple", headers=headers)
            assert resp.status_code == 200
            
            # Test Timeline
            resp = client.post("/get_timeline", json={"entity_id": "Steve Jobs"}, headers=headers)
            assert resp.status_code == 200
            
            # Test Stats
            resp = client.get("/stats", headers=headers)
            assert resp.status_code == 200
            
            # Test Compact
            resp = client.post("/compact", headers=headers)
            assert resp.status_code == 200
            
    finally:
        # Restore original globals
        server.NODE_MODE = orig_mode
        server.shard_nodes = orig_shards
        server.client = orig_client
        server.hash_ring = orig_hash_ring
        server.API_KEY = orig_api_key

def test_visualize_endpoint():
    """Verify that the /visualize dashboard returns a valid HTML response with 3D Force Graph imports."""
    with TestClient(app) as client:
        resp = client.get("/visualize")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "EpochDB" in resp.text
        assert "3d-force-graph" in resp.text.lower()

