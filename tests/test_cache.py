import os
import sys
import tempfile
import shutil
import pytest
from fastapi.testclient import TestClient

# Setup environment variables BEFORE importing app
temp_dir = tempfile.mkdtemp(prefix="epochdb_cache_test_")
os.environ["NODE_MODE"] = "shard"
os.environ["INTERNAL_AUTH_TOKEN"] = "test-token-12345"
os.environ["STORAGE_DIR"] = temp_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.server import app, CLUSTER_STATE_VERSIONS, LOCAL_READ_CACHE

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_cache_etag_flow():
    headers = {"X-Internal-Token": "test-token-12345"}
    
    with TestClient(app) as client:
        # Clear cache structures to isolate this test
        CLUSTER_STATE_VERSIONS.clear()
        LOCAL_READ_CACHE.clear()

        # 1. /remember some data
        resp = client.post("/remember", json={"text": "EpochDB caching is awesome."}, headers=headers)
        assert resp.status_code == 201
        mem_id = resp.json()["id"]

        # 2. First /get should return ETag header
        resp = client.post("/get", json={"memory_id": mem_id}, headers=headers)
        assert resp.status_code == 200
        etag1 = resp.headers.get("etag")
        assert etag1 is not None

        # 3. Second /get with If-None-Match should return 304 Not Modified
        headers_with_etag = {**headers, "If-None-Match": etag1}
        resp = client.post("/get", json={"memory_id": mem_id}, headers=headers_with_etag)
        assert resp.status_code == 304
        assert resp.text == ""

        # 4. /query should also generate ETag and cache
        query_payload = {"query": "EpochDB caching", "k": 3}
        resp = client.post("/query", json=query_payload, headers=headers)
        assert resp.status_code == 200
        etag_query = resp.headers.get("etag")
        assert etag_query is not None

        # 5. Query validation with If-None-Match -> 304
        headers_query_etag = {**headers, "If-None-Match": etag_query}
        resp = client.post("/query", json=query_payload, headers=headers_query_etag)
        assert resp.status_code == 304

        # 6. Mutate database (write something else) -> invalidates cache
        resp = client.post("/remember", json={"text": "Another independent fact."}, headers=headers)
        assert resp.status_code == 201

        # 7. Query again -> should be cache miss (status 200 instead of 304), returning a new ETag
        resp = client.post("/query", json=query_payload, headers=headers_query_etag)
        assert resp.status_code == 200
        etag_query2 = resp.headers.get("etag")
        assert etag_query2 != etag_query

def test_tenant_namespace_isolation():
    # Test that caching is isolated to tenant and namespace
    headers_t1_ns1 = {
        "X-Internal-Token": "test-token-12345",
        "X-Tenant": "tenant1",
        "X-Namespace": "ns1"
    }
    headers_t1_ns2 = {
        "X-Internal-Token": "test-token-12345",
        "X-Tenant": "tenant1",
        "X-Namespace": "ns2"
    }

    with TestClient(app) as client:
        # Clear cache structures
        CLUSTER_STATE_VERSIONS.clear()
        LOCAL_READ_CACHE.clear()

        # Ingest into tenant1/ns1
        resp = client.post("/remember", json={"text": "Tenant 1 Namespace 1 memory"}, headers=headers_t1_ns1)
        assert resp.status_code == 201
        mem_id = resp.json()["id"]

        # Cache a query for tenant1/ns1
        query_payload = {"query": "Tenant 1", "k": 2}
        resp = client.post("/query", json=query_payload, headers=headers_t1_ns1)
        assert resp.status_code == 200
        etag_t1_ns1 = resp.headers.get("etag")

        # Cache a query for tenant1/ns2
        resp = client.post("/query", json=query_payload, headers=headers_t1_ns2)
        assert resp.status_code == 200
        etag_t1_ns2 = resp.headers.get("etag")

        # Verify If-None-Match works for both
        resp = client.post("/query", json=query_payload, headers={**headers_t1_ns1, "If-None-Match": etag_t1_ns1})
        assert resp.status_code == 304
        resp = client.post("/query", json=query_payload, headers={**headers_t1_ns2, "If-None-Match": etag_t1_ns2})
        assert resp.status_code == 304

        # Ingest into tenant1/ns1 -> should invalidate tenant1/ns1 cache only
        client.post("/remember", json={"text": "New Tenant 1 Namespace 1 memory"}, headers=headers_t1_ns1)

        # tenant1/ns1 query should NOT hit cache (status 200, new etag)
        resp = client.post("/query", json=query_payload, headers={**headers_t1_ns1, "If-None-Match": etag_t1_ns1})
        assert resp.status_code == 200

        # tenant1/ns2 query should STILL hit cache (status 304)
        resp = client.post("/query", json=query_payload, headers={**headers_t1_ns2, "If-None-Match": etag_t1_ns2})
        assert resp.status_code == 304
