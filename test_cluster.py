import os
import sys
import time
import shutil
import tempfile
import subprocess
import asyncio
import httpx
import pytest
from client import AsyncRemoteEpochDB

# Temporary storage directories for shards
SHARD0_DIR = tempfile.mkdtemp(prefix="epochdb_shard0_")
SHARD1_DIR = tempfile.mkdtemp(prefix="epochdb_shard1_")
SHARD2_DIR = tempfile.mkdtemp(prefix="epochdb_shard2_")

@pytest.fixture(scope="module", autouse=True)
def run_cluster():
    processes = []
    
    # 1. Start Shard 0
    env0 = os.environ.copy()
    env0["NODE_MODE"] = "shard"
    env0["STORAGE_DIR"] = SHARD0_DIR
    p0 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", "8081"],
        env=env0, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p0)

    # 2. Start Shard 1
    env1 = os.environ.copy()
    env1["NODE_MODE"] = "shard"
    env1["STORAGE_DIR"] = SHARD1_DIR
    p1 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", "8082"],
        env=env1, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p1)

    # 3. Start Shard 2
    env2 = os.environ.copy()
    env2["NODE_MODE"] = "shard"
    env2["STORAGE_DIR"] = SHARD2_DIR
    p2 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", "8083"],
        env=env2, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p2)

    # 4. Start Coordinator Gateway
    env_coord = os.environ.copy()
    env_coord["NODE_MODE"] = "coordinator"
    env_coord["SHARD_NODES"] = "http://127.0.0.1:8081,http://127.0.0.1:8082,http://127.0.0.1:8083"
    p_coord = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", "8080"],
        env=env_coord, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p_coord)

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
            resp_coord = httpx.get("http://127.0.0.1:8080/stats", timeout=1.0)
            resp_s0 = httpx.get("http://127.0.0.1:8081/stats", timeout=1.0)
            resp_s1 = httpx.get("http://127.0.0.1:8082/stats", timeout=1.0)
            resp_s2 = httpx.get("http://127.0.0.1:8083/stats", timeout=1.0)
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
        pytest.fail("Cluster coordinator failed to respond on /stats within 30 seconds.")

    yield

    # Clean up subprocesses
    for p in processes:
        p.terminate()
        p.wait()

    # Clean up temporary directories
    for d in [SHARD0_DIR, SHARD1_DIR, SHARD2_DIR]:
        shutil.rmtree(d, ignore_errors=True)

@pytest.mark.asyncio
async def test_cluster_operations():
    # Initialize clients for Coordinator and Shards
    coord_db = AsyncRemoteEpochDB(host="127.0.0.1", port=8080)
    shard0 = AsyncRemoteEpochDB(host="127.0.0.1", port=8081)
    shard1 = AsyncRemoteEpochDB(host="127.0.0.1", port=8082)
    shard2 = AsyncRemoteEpochDB(host="127.0.0.1", port=8083)

    try:
        # 1. Ingest memories through coordinator
        m1 = "Albert Einstein was a theoretical physicist."
        m2 = "Marie Curie discovered radium and polonium."
        m3 = "Isaac Newton formulated the laws of gravity."

        r1 = await coord_db.remember(m1, metadata={"triples": [("Albert Einstein", "is_a", "physicist")]})
        r2 = await coord_db.remember(m2, metadata={"triples": [("Marie Curie", "discovered", "radium")]})
        r3 = await coord_db.remember(m3, metadata={"triples": [("Isaac Newton", "formulated", "gravity")]})

        id1 = r1["id"]
        id2 = r2["id"]
        id3 = r3["id"]

        # Assert ID prefix matches standard pattern
        assert id1.startswith("shard")
        assert id2.startswith("shard")
        assert id3.startswith("shard")

        shards = [shard0, shard1, shard2]
        
        def get_shard_index(mem_id):
            return int(mem_id.split("-")[0][5:])

        # 2. Verify memories were correctly sharded to target nodes
        idx1 = get_shard_index(id1)
        m1_shard_data = await shards[idx1].get(id1)
        assert m1_shard_data["payload"] == m1

        idx2 = get_shard_index(id2)
        m2_shard_data = await shards[idx2].get(id2)
        assert m2_shard_data["payload"] == m2

        idx3 = get_shard_index(id3)
        m3_shard_data = await shards[idx3].get(id3)
        assert m3_shard_data["payload"] == m3

        # 3. Query through coordinator (Re-ranking & Merging)
        query_results = await coord_db.query("Newton formulated gravity physics", k=3)
        assert len(query_results) > 0
        # The Newton memory should be returned
        assert any("Newton" in r["text"] for r in query_results)
        
        # Verify scores are returned
        for r in query_results:
            assert "score" in r
            assert isinstance(r["score"], float)

        # 4. Fetch entity graph from coordinator (distributed merge)
        graph = await coord_db.entity_graph("Marie Curie")
        assert "Marie Curie" in graph["nodes"]
        assert len(graph["edges"]) > 0

        # 5. Point query GET via coordinator (direct routing)
        point_mem = await coord_db.get(id2)
        assert point_mem["payload"] == m2

        # 6. Update memory via coordinator (direct routing)
        await coord_db.update(id3, text="Isaac Newton formulated the laws of gravity and calculus.")
        updated_mem = await coord_db.get(id3)
        assert "calculus" in updated_mem["payload"]

        # 7. Get timeline across the cluster (merged chronologically)
        timeline = await coord_db.get_timeline(entity_id="Isaac Newton")
        assert len(timeline) > 0
        assert "calculus" in timeline[0]["payload"]

        # 8. Stats check (merged counts)
        stats = await coord_db.stats()
        assert stats["memory_count"] == 3

        # 9. Delete memory (direct routing)
        await coord_db.delete(id1, hard=True)
        deleted_mem = await coord_db.get(id1)
        assert not deleted_mem or "_deleted" in deleted_mem.get("metadata", {})

        # 10. Check stats again
        stats2 = await coord_db.stats()
        assert stats2["memory_count"] == 2

    finally:
        await coord_db.close()
        await shard0.close()
        await shard1.close()
        await shard2.close()
