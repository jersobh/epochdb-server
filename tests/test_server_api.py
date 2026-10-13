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
from src.server import app, SERVER_VERSION

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    shutil.rmtree(temp_dir, ignore_errors=True)

import httpx

class MockResponse(httpx.Response):
    def __init__(self, status_code, json_data):
        super().__init__(status_code, json=json_data)

class MockAsyncClient:
    async def post(self, url, **kwargs):
        if "/remember_batch" in url:
            items = kwargs.get("json", {}).get("items", [])
            ids = [it.get("id", f"test-id-{i}") for i, it in enumerate(items)] if items else ["test-batch-1"]
            return MockResponse(201, {"status": "success", "atom_ids": ids})
        elif "/remember" in url:
            return MockResponse(201, {"status": "success", "id": kwargs.get("json", {}).get("id", "test-id")})
        elif "/replace" in url:
            return MockResponse(200, {"status": "success", "id": kwargs.get("json", {}).get("atom_id", "test-id")})
        elif "/propose" in url:
            return MockResponse(200, {"status": "APPROVED", "memory_id": kwargs.get("json", {}).get("id", "test-prop-id")})
        elif "/get" in url:
            return MockResponse(200, {"id": "test-id", "payload": "Water boils at 100 degrees Celsius."})
        elif "/query" in url:
            return MockResponse(200, {"results": [{"id": "test-id", "score": 0.95}]})
        elif "/multi_hop" in url:
            return MockResponse(200, {"results": [{"id": "test-id", "score": 0.95}], "memories": [{"id": "test-id", "score": 0.95}]})
        elif "/skills/remember" in url:
            return MockResponse(201, {"status": "success", "id": kwargs.get("json", {}).get("skill_id", "skill-123"), "skill_name": kwargs.get("json", {}).get("skill_name", "test_skill")})
        elif "/skills/get" in url:
            return MockResponse(200, {"id": "skill-123", "payload": "Test Skill Payload"})
        elif "/profiles/remember" in url:
            return MockResponse(201, {"status": "success", "id": "prof-123", "user_id": kwargs.get("json", {}).get("user_id", "user1")})
        elif "/profiles/get" in url:
            return MockResponse(200, {"memories": [{"id": "prof-123", "payload": "Prefers dark mode"}]})
        elif "/summary/snapshot" in url:
            return MockResponse(200, {"snapshot": "Hot memory snapshot summary"})
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
        elif "/skills/" in url:
            return MockResponse(200, {"id": "skill-123", "payload": "Test Skill Payload"})
        elif "/skills" in url:
            return MockResponse(200, {"skills": [{"id": "skill-123", "payload": "Test Skill Payload"}]})
        elif "/profiles/" in url:
            return MockResponse(200, {"memories": [{"id": "prof-123", "payload": "Prefers dark mode"}]})
        elif "/summary/snapshot" in url:
            return MockResponse(200, {"snapshot": "Hot memory snapshot summary"})
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
        assert resp.json() == {"status": "healthy", "mode": "shard", "version": SERVER_VERSION}

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
            assert resp.json() == {"status": "healthy", "mode": "coordinator", "version": SERVER_VERSION}
            
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
    """Verify that /visualize serves the React Studio SPA with injected server config."""
    with TestClient(app) as client:
        resp = client.get("/visualize")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "EpochDB" in resp.text
        assert "id=\"root\"" in resp.text or "id='root'" in resp.text
        assert "__EPOCHDB_SERVER_CONFIG__" in resp.text
        assert "./assets/" in resp.text or "/assets/" in resp.text


def test_analytics_query_endpoint():
    """Verify that POST /v1/analytics/query executes DuckDB queries over Cold Tier Parquet archives."""
    with TestClient(app) as client:
        headers = {"X-API-Key": "test-token-12345"}
        resp = client.post("/remember", json={"text": "Water boils at 100 degrees Celsius.", "triples": [["Water", "boils_at", "100C"]]}, headers=headers)
        assert resp.status_code == 201

        resp = client.post("/compact", headers=headers)
        assert resp.status_code == 200

        resp = client.post("/v1/analytics/query", json={"query": "SELECT COUNT(*) as count FROM cold_tier"}, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "data" in data

def test_remember_batch():
    """Verify batch ingestion of multiple memories (/remember_batch)."""
    headers = {"X-Internal-Token": "test-token-12345"}
    with TestClient(app) as client:
        batch_items = [
            {"text": "Python was created by Guido van Rossum.", "metadata": {"topic": "python"}},
            {"text": "Rust guarantees memory safety without garbage collection.", "metadata": {"topic": "rust"}},
            {"text": "Go was designed at Google.", "metadata": {"topic": "go"}}
        ]
        resp = client.post("/remember_batch", json={"items": batch_items}, headers=headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "success"
        assert "atom_ids" in data
        assert len(data["atom_ids"]) == 3

        # Verify we can query one of the batch ingested memories
        q_resp = client.post("/query", json={"query": "Guido van Rossum", "k": 1}, headers=headers)
        assert q_resp.status_code == 200
        q_data = q_resp.json()
        assert len(q_data["results"]) == 1
        assert "Python" in q_data["results"][0]["text"]

def test_replace_memory():
    """Verify in-place durable replacement of an existing memory (/replace)."""
    headers = {"X-Internal-Token": "test-token-12345"}
    with TestClient(app) as client:
        # First ingest a memory
        resp = client.post("/remember", json={"text": "Original text before replacement."}, headers=headers)
        assert resp.status_code == 201
        atom_id = resp.json()["id"]

        # Now replace it in-place
        rep_resp = client.post("/replace", json={"atom_id": atom_id, "text": "Replaced text with updated payload."}, headers=headers)
        assert rep_resp.status_code == 200
        assert rep_resp.json()["status"] == "success"

        # Verify retrieval returns the new text
        get_resp = client.post("/get", json={"memory_id": atom_id}, headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["payload"] == "Replaced text with updated payload."

def test_multi_hop_retrieval():
    """Verify multi-hop relational search across the knowledge graph."""
    headers = {"X-Internal-Token": "test-token-12345"}
    with TestClient(app) as client:
        # Ingest connected entity triples
        client.post("/remember", json={"text": "Marie Curie discovered Radium.", "metadata": {"triples": [["Marie Curie", "discovered", "Radium"]]}, "id": "mhop-1"}, headers=headers)
        client.post("/remember", json={"text": "Radium is used in Radiation Therapy.", "metadata": {"triples": [["Radium", "used_in", "Radiation Therapy"]]}, "id": "mhop-2"}, headers=headers)

        resp = client.post("/multi_hop", json={"query": "Marie Curie", "hops": 2, "k": 5}, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "memories" in data
        assert len(data["results"]) >= 1

def test_skills_lifecycle():
    """Verify procedural skill registration, retrieval, and listing."""
    headers = {"X-Internal-Token": "test-token-12345"}
    with TestClient(app) as client:
        skill_payload = {
            "skill_name": "data_analysis",
            "description": "Analyze tabular dataset using DuckDB and generate summary",
            "steps": [
                {"step": 1, "action": "load_parquet"},
                {"step": 2, "action": "run_aggregations"},
                {"step": 3, "action": "render_chart"}
            ],
            "tool_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            "decision_rules": ["If file is csv, convert to parquet first"]
        }
        # 1. Register skill
        reg_resp = client.post("/skills/remember", json=skill_payload, headers=headers)
        assert reg_resp.status_code == 201
        reg_data = reg_resp.json()
        assert reg_data["status"] == "success"
        assert reg_data["skill_name"] == "data_analysis"

        # 2. Get skill by name via GET
        get_resp = client.get("/skills/data_analysis", headers=headers)
        assert get_resp.status_code == 200
        skill_data = get_resp.json()
        assert "data_analysis" in str(skill_data)

        # 3. Get skill via POST /skills/get
        post_get_resp = client.post("/skills/get", json={"skill_name": "data_analysis"}, headers=headers)
        assert post_get_resp.status_code == 200
        assert "data_analysis" in str(post_get_resp.json())

        # 4. List skills
        list_resp = client.get("/skills", headers=headers)
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert "skills" in list_data
        assert any("data_analysis" in str(s) for s in list_data["skills"])

def test_profiles_and_summary_lifecycle():
    """Verify user profile facts storage, retrieval, and summary snapshot."""
    headers = {"X-Internal-Token": "test-token-12345"}
    with TestClient(app) as client:
        # 1. Store profile fact
        prof_resp = client.post("/profiles/remember", json={
            "user_id": "user_42",
            "fact_text": "Prefers concise code reviews in Python 3.12."
        }, headers=headers)
        assert prof_resp.status_code == 201
        assert prof_resp.json()["user_id"] == "user_42"

        # 2. Retrieve user profile via GET /profiles/{user_id}
        get_resp = client.get("/profiles/user_42", headers=headers)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert "memories" in data
        assert len(data["memories"]) >= 1

        # 3. Retrieve user profile via POST /profiles/get
        post_get = client.post("/profiles/get", json={"user_id": "user_42"}, headers=headers)
        assert post_get.status_code == 200
        assert len(post_get.json()["memories"]) >= 1

        # 4. Generate summary snapshot
        snap_resp = client.get("/summary/snapshot?user_id=user_42", headers=headers)
        assert snap_resp.status_code == 200
        assert "snapshot" in snap_resp.json()

def test_propose_validation():
    """Verify two-phase state commit (Neuro-Symbolic LCAG propose)."""
    headers = {"X-Internal-Token": "test-token-12345"}
    with TestClient(app) as client:
        resp = client.post("/propose", json={
            "text": "Temperature sensor calibrated to 25.0 C."
        }, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("APPROVED", "REJECTED")
        if data["status"] == "APPROVED":
            assert "memory_id" in data

def test_triplet_ingestion_and_graph_linking():
    """Verify direct triplet ingestion without text and entity graph node linking."""
    headers = {"X-Internal-Token": "test-token-12345"}
    with TestClient(app) as client:
        # Ingest direct subject-predicate-object triplet without providing text
        resp = client.post("/remember", json={
            "subject": "Nikola Tesla",
            "predicate": "invented",
            "object": "AC Motor"
        }, headers=headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "success"

        # Ingest multi-triplet array without providing text
        resp2 = client.post("/remember", json={
            "triples": [
                ["AC Motor", "powered", "Modern Grid"],
                ["Nikola Tesla", "born_in", "Smiljan"]
            ]
        }, headers=headers)
        assert resp2.status_code == 201

        # Query entity graph
        graph_resp = client.get("/entity_graph?entity_id=Nikola Tesla&depth=2", headers=headers)
        assert graph_resp.status_code == 200
        graph = graph_resp.json()
        assert "Nikola Tesla" in graph["nodes"]
        assert "AC Motor" in graph["nodes"]


def test_granular_export_and_import():
    """Verify granular export of skills, profiles, and triplets, and re-importing into another namespace."""
    headers = {"X-Internal-Token": "test-token-12345", "X-Namespace": "source-ns"}
    dest_headers = {"X-Internal-Token": "test-token-12345", "X-Namespace": "target-ns"}

    with TestClient(app) as client:
        # 1. Ingest procedural skill
        skill_resp = client.post("/skills/remember", json={
            "skill_name": "kubernetes_pod_restart",
            "description": "Procedure to restart a crashing kubernetes pod",
            "steps": ["kubectl get pods", "kubectl rollout restart deployment/<name>"],
            "tool_schema": {"type": "object", "properties": {"name": {"type": "string"}}}
        }, headers=headers)
        assert skill_resp.status_code == 201

        # 2. Ingest user profile
        prof_resp = client.post("/profiles/remember", json={
            "user_id": "alice_dev",
            "fact_text": "Prefers dark mode and vim keybindings."
        }, headers=headers)
        assert prof_resp.status_code == 201

        # 3. Ingest knowledge graph triplet
        trip_resp = client.post("/remember", json={
            "subject": "Quantum Computing",
            "predicate": "relies_on",
            "object": "Superposition"
        }, headers=headers)
        assert trip_resp.status_code == 201

        # 4. Export granularly: skills
        exp_skills = client.get("/export?data_type=skills", headers=headers)
        assert exp_skills.status_code == 200
        skills_bundle = exp_skills.json()
        assert skills_bundle["counts"]["skills"] >= 1
        assert any(
            (s.get("metadata", {}).get("skill_name") == "kubernetes_pod_restart" or s.get("id") == "kubernetes_pod_restart")
            for s in skills_bundle["data"]["skills"]
        )

        # 5. Export granularly: triplets
        exp_triplets = client.get("/export?data_type=triplets", headers=headers)
        assert exp_triplets.status_code == 200
        triplets_bundle = exp_triplets.json()
        assert triplets_bundle["counts"]["triplets"] >= 1
        assert any(t[0] == "Quantum Computing" and t[2] == "Superposition" for t in triplets_bundle["data"]["triplets"])

        # 6. Export all
        exp_all = client.get("/export?data_type=all", headers=headers)
        assert exp_all.status_code == 200
        all_bundle = exp_all.json()
        assert all_bundle["counts"]["skills"] >= 1
        assert all_bundle["counts"]["profiles"] >= 1
        assert all_bundle["counts"]["triplets"] >= 1

        # 7. Import bundle into target-ns
        import_resp = client.post("/import", json={
            "data_type": "all",
            "data": all_bundle
        }, headers=dest_headers)
        assert import_resp.status_code == 201
        imported_stats = import_resp.json()
        assert imported_stats["status"] == "success"
        assert imported_stats["imported"]["skills"] >= 1

        # 8. Verify destination namespace has the skill
        list_skills_resp = client.get("/skills", headers=dest_headers)
        assert list_skills_resp.status_code == 200
        dest_skills = list_skills_resp.json().get("skills", [])
        assert any(
            (s.get("metadata", {}).get("skill_name") == "kubernetes_pod_restart" or s.get("id") == "kubernetes_pod_restart")
            for s in dest_skills
        )





