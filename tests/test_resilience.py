import os
import sys
import time
import shutil
import tempfile
import subprocess
import asyncio
import httpx
import pytest

# Add parent directory to path to resolve client import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from epochdb import AsyncRemoteEpochDB
from src.server import ConsistentHashRing



# Temporary storage directories for shards
SHARD0_DIR = tempfile.mkdtemp(prefix="epochdb_res_shard0_")
SHARD1_DIR = tempfile.mkdtemp(prefix="epochdb_res_shard1_")
SHARD2_DIR = tempfile.mkdtemp(prefix="epochdb_res_shard2_")

PORT_COORD = 28080
PORT_S0 = 28081
PORT_S1 = 28082
PORT_S2 = 28083

# Local hash ring for test mapping predictions
TEST_SHARD_NODES = [f"http://127.0.0.1:{PORT_S0}", f"http://127.0.0.1:{PORT_S1}", f"http://127.0.0.1:{PORT_S2}"]
test_hash_ring = ConsistentHashRing(TEST_SHARD_NODES)


# Global reference to processes so they can be managed inside tests
cluster_processes = []
cluster_envs = []

@pytest.fixture(scope="module", autouse=True)
def run_cluster():
    global cluster_processes, cluster_envs
    processes = []
    
    # Common internal token and API key
    api_key = "test-api-key-12345"
    internal_token = "test-internal-token-67890"

    # 1. Start Shard 0
    env0 = os.environ.copy()
    env0["NODE_MODE"] = "shard"
    env0["STORAGE_DIR"] = SHARD0_DIR
    env0["INTERNAL_AUTH_TOKEN"] = internal_token
    p0 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S0)],
        env=env0, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p0)
    cluster_envs.append(env0)

    # 2. Start Shard 1
    env1 = os.environ.copy()
    env1["NODE_MODE"] = "shard"
    env1["STORAGE_DIR"] = SHARD1_DIR
    env1["INTERNAL_AUTH_TOKEN"] = internal_token
    p1 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S1)],
        env=env1, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p1)
    cluster_envs.append(env1)

    # 3. Start Shard 2
    env2 = os.environ.copy()
    env2["NODE_MODE"] = "shard"
    env2["STORAGE_DIR"] = SHARD2_DIR
    env2["INTERNAL_AUTH_TOKEN"] = internal_token
    p2 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S2)],
        env=env2, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p2)
    cluster_envs.append(env2)

    # 4. Start Coordinator Gateway
    env_coord = os.environ.copy()
    env_coord["NODE_MODE"] = "coordinator"
    env_coord["SHARD_NODES"] = f"http://127.0.0.1:{PORT_S0},http://127.0.0.1:{PORT_S1},http://127.0.0.1:{PORT_S2}"
    env_coord["API_KEY"] = api_key
    env_coord["INTERNAL_AUTH_TOKEN"] = internal_token
    p_coord = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_COORD)],
        env=env_coord, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p_coord)
    cluster_envs.append(env_coord)

    cluster_processes = processes

    # Wait for coordinator and shards to become healthy
    start_time = time.time()
    healthy = False
    while time.time() - start_time < 30:
        # Check if any process exited prematurely
        for p in processes:
            if p.poll() is not None:
                print(f"Process exited prematurely: {p.args} with code {p.returncode}", file=sys.stderr)
                pytest.fail(f"Server subprocess {p.args} failed to start.")
                
        try:
            resp_coord = httpx.get(f"http://127.0.0.1:{PORT_COORD}/healthz", timeout=1.0)
            resp_s0 = httpx.get(f"http://127.0.0.1:{PORT_S0}/healthz", timeout=1.0)
            resp_s1 = httpx.get(f"http://127.0.0.1:{PORT_S1}/healthz", timeout=1.0)
            resp_s2 = httpx.get(f"http://127.0.0.1:{PORT_S2}/healthz", timeout=1.0)
            if (resp_coord.status_code == 200 and 
                resp_s0.status_code == 200 and 
                resp_s1.status_code == 200 and 
                resp_s2.status_code == 200):
                healthy = True
                break
        except Exception:
            pass
        time.sleep(1.0)

    if not healthy:
        for p in processes:
            p.terminate()
            p.wait()
        pytest.fail("Cluster coordinator failed to respond on /healthz within 30 seconds.")

    yield

    # Clean up subprocesses
    for p in cluster_processes:
        if p.poll() is None:
            p.terminate()
            p.wait()

    # Clean up temporary directories
    for d in [SHARD0_DIR, SHARD1_DIR, SHARD2_DIR]:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.asyncio
async def test_security_authorization():
    """
    Validates that correct keys authorize clients and missing/invalid keys fail with 401.
    """
    # 1. Access Coordinator without key
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{PORT_COORD}/stats")
        assert resp.status_code == 401
        
        # Access with wrong key
        resp = await client.get(f"http://127.0.0.1:{PORT_COORD}/stats", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

        # Access with correct key
        resp = await client.get(f"http://127.0.0.1:{PORT_COORD}/stats", headers={"X-API-Key": "test-api-key-12345"})
        assert resp.status_code == 200

    # 2. Access Shard directly
    async with httpx.AsyncClient() as client:
        # Shards should reject regular clients without internal auth token
        resp = await client.post(f"http://127.0.0.1:{PORT_S0}/query", json={"query": "test", "k": 1})
        assert resp.status_code == 401
        
        # Wrong token
        resp = await client.post(f"http://127.0.0.1:{PORT_S0}/query", json={"query": "test", "k": 1}, headers={"X-Internal-Token": "bad-token"})
        assert resp.status_code == 401

        # Correct token
        resp = await client.post(f"http://127.0.0.1:{PORT_S0}/query", json={"query": "test", "k": 1}, headers={"X-Internal-Token": "test-internal-token-67890"})
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_high_concurrency_stress():
    """
    Verifies that the server and SQLite/HNSW storage engine handle concurrent execution 
    without throwing file lock exceptions or dropped requests.
    """
    coord_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_COORD, api_key="test-api-key-12345")
    
    try:
        # Generate 50 concurrent writes
        texts = [f"This is concurrent memory atom number {i} for stress testing database lock integrity." for i in range(50)]
        
        # Dispatch concurrently
        tasks = [coord_db.remember(text, metadata={"index": i, "triples": [("StressTest", "run", f"num_{i}")]}) for i, text in enumerate(texts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all operations completed successfully without throwing exceptions
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                pytest.fail(f"Concurrent write {idx} failed with exception: {res}")
            assert isinstance(res, str)
            assert res.startswith("shard")

        # Verify coordinator stats shows correct count
        stats = await coord_db.stats()
        assert stats["memory_count"] >= 50
        
    finally:
        await coord_db.close()


@pytest.mark.asyncio
async def test_shard_failure_and_graceful_recovery():
    """
    Tests fault tolerance: kills a shard, verifies coordinator continues processing reads,
    re-spins the shard, and verifies complete recovery.
    """
    coord_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_COORD, api_key="test-api-key-12345")
    shard0_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_S0, api_key="test-internal-token-67890")
    shard1_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_S1, api_key="test-internal-token-67890")
    shard2_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_S2, api_key="test-internal-token-67890")

    try:
        # Ingest a memory onto each shard
        # Since we use consistent hashing, we'll write multiple items and determine which goes where by prefix
        ids_by_shard = {0: [], 1: [], 2: []}
        
        for i in range(30):
            text = f"Resilience test content identifier {i}"
            res = await coord_db.remember(text, metadata={"type": "resilience", "triples": [("Resilience", "value", str(i))]})
            mem_id = res
            prefix = mem_id.split("-")[0]
            shard_idx = int(prefix[5:])
            ids_by_shard[shard_idx].append(mem_id)

        # Confirm we have at least one ID for Shard 1 (the one we will kill)
        assert len(ids_by_shard[1]) > 0
        assert len(ids_by_shard[0]) > 0
        assert len(ids_by_shard[2]) > 0

        target_shard_1_id = ids_by_shard[1][0]
        target_shard_0_id = ids_by_shard[0][0]

        # ----------------------------------------------------
        # Simulate Shard 1 Outage (terminate its process)
        # ----------------------------------------------------
        p1 = cluster_processes[1]
        p1.terminate()
        p1.wait()

        # 1. Point GET directly routed to active Shard 0 should succeed
        res_active = await coord_db.get(target_shard_0_id)
        assert res_active.id == target_shard_0_id

        # 2. Point GET directly routed to offline Shard 1 should fail with a clean RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await coord_db.get(target_shard_1_id)
        assert "HTTP Error 500" in str(exc_info.value) or "Failed to connect" in str(exc_info.value)

        # 3. Vector query fanning out to all shards should degrade gracefully:
        # Shard 1 is down, but Shard 0 & 2 are up. The coordinator should return results from healthy shards.
        query_results = await coord_db.query("Resilience test content query", k=10)
        assert len(query_results) > 0
        for r in query_results:
            # None of the results should be from Shard 1
            assert not r.id.startswith("shard1-")

        # 4. Timeline query (broadcast) should degrade gracefully
        timeline = await coord_db.get_timeline("Resilience")
        assert len(timeline) > 0
        for item in timeline:
            assert not item.id.startswith("shard1-")

        # 5. Entity graph (broadcast) should degrade gracefully
        graph = await coord_db.entity_graph("Resilience")
        assert len(graph["nodes"]) > 0

        # 6. Database Stats should succeed, returning aggregated count of healthy shards
        stats = await coord_db.stats()
        assert stats["memory_count"] > 0

        # 7. Write mapped to active Shard 0 succeeds
        # Find a text that hashes to Shard 0 or Shard 2
        success_write = False
        for j in range(100):
            test_txt = f"Verify active write {j}"
            # Predict node
            node = test_hash_ring.get_node(test_txt)
            if node == f"http://127.0.0.1:{PORT_S0}" or node == f"http://127.0.0.1:{PORT_S2}":
                res = await coord_db.remember(test_txt)
                assert isinstance(res, str)
                success_write = True
                break
        assert success_write

        # 8. Write mapped to offline Shard 1 fails with 500
        fail_write = False
        for j in range(100):
            test_txt = f"Verify offline write {j}"
            node = test_hash_ring.get_node(test_txt)
            if node == f"http://127.0.0.1:{PORT_S1}":
                with pytest.raises(RuntimeError) as exc_info:
                    await coord_db.remember(test_txt)
                assert "HTTP Error 500" in str(exc_info.value) or "Failed to connect" in str(exc_info.value)
                fail_write = True
                break
        assert fail_write

        # ----------------------------------------------------
        # Recovery phase: Restart Shard 1
        # ----------------------------------------------------
        # Re-initialize Shard 1 using the same storage directory & env
        p1_recovered = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S1)],
            env=cluster_envs[1], stdout=sys.stderr, stderr=sys.stderr
        )
        cluster_processes[1] = p1_recovered

        # Wait for the recovered Shard 1 to become healthy
        start_time = time.time()
        recovered = False
        while time.time() - start_time < 30:
            try:
                resp = httpx.get(f"http://127.0.0.1:{PORT_S1}/healthz", timeout=1.0)
                if resp.status_code == 200:
                    recovered = True
                    break
            except Exception:
                pass
            time.sleep(1.0)

        assert recovered, "Recovered shard failed to start within 30 seconds."

        # 9. Direct point GET to the recovered Shard 1 now succeeds
        res_recovered = await coord_db.get(target_shard_1_id)
        assert res_recovered.id == target_shard_1_id

        # 10. Write mapped to recovered Shard 1 now succeeds
        recovered_write = False
        for j in range(100):
            test_txt = f"Recovered write {j}"
            node = test_hash_ring.get_node(test_txt)
            if node == f"http://127.0.0.1:{PORT_S1}":
                res = await coord_db.remember(test_txt)
                assert isinstance(res, str)
                recovered_write = True
                break
        assert recovered_write

    finally:
        await coord_db.close()
        await shard0_db.close()
        await shard1_db.close()
        await shard2_db.close()
