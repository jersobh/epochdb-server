import os
import sys
import time
import shutil
import tempfile
import subprocess
import asyncio
import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from epochdb import AsyncRemoteEpochDB
from src.server import ConsistentHashRing

# Temporary storage directories for replicas
SHARD0_A_DIR = tempfile.mkdtemp(prefix="epochdb_rep_shard0_a_")
SHARD0_B_DIR = tempfile.mkdtemp(prefix="epochdb_rep_shard0_b_")

PORT_COORD = 29080
PORT_S0_A = 29081
PORT_S0_B = 29082

cluster_processes = []
cluster_envs = []

@pytest.fixture(scope="module", autouse=True)
def run_replica_cluster():
    global cluster_processes, cluster_envs
    processes = []
    
    api_key = "test-api-key-12345"
    internal_token = "test-internal-token-67890"

    # 1. Start Shard 0 - Replica A
    env_a = os.environ.copy()
    env_a["NODE_MODE"] = "shard"
    env_a["STORAGE_DIR"] = SHARD0_A_DIR
    env_a["INTERNAL_AUTH_TOKEN"] = internal_token
    p_a = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S0_A)],
        env=env_a, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p_a)
    cluster_envs.append(env_a)

    # 2. Start Shard 0 - Replica B
    env_b = os.environ.copy()
    env_b["NODE_MODE"] = "shard"
    env_b["STORAGE_DIR"] = SHARD0_B_DIR
    env_b["INTERNAL_AUTH_TOKEN"] = internal_token
    p_b = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S0_B)],
        env=env_b, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p_b)
    cluster_envs.append(env_b)

    # 3. Start Coordinator
    env_coord = os.environ.copy()
    env_coord["NODE_MODE"] = "coordinator"
    # Specify replica grouping using +
    env_coord["SHARD_NODES"] = f"http://127.0.0.1:{PORT_S0_A}+http://127.0.0.1:{PORT_S0_B}"
    env_coord["API_KEY"] = api_key
    env_coord["INTERNAL_AUTH_TOKEN"] = internal_token
    p_coord = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_COORD)],
        env=env_coord, stdout=sys.stderr, stderr=sys.stderr
    )
    processes.append(p_coord)
    cluster_envs.append(env_coord)

    cluster_processes = processes

    # Wait for nodes to become healthy
    start_time = time.time()
    healthy = False
    while time.time() - start_time < 120:
        for p in processes:
            if p.poll() is not None:
                pytest.fail(f"Subprocess failed to start with code {p.returncode}")
        try:
            resp_coord = httpx.get(f"http://127.0.0.1:{PORT_COORD}/healthz", timeout=1.0)
            resp_a = httpx.get(f"http://127.0.0.1:{PORT_S0_A}/healthz", timeout=1.0)
            resp_b = httpx.get(f"http://127.0.0.1:{PORT_S0_B}/healthz", timeout=1.0)
            if resp_coord.status_code == 200 and resp_a.status_code == 200 and resp_b.status_code == 200:
                healthy = True
                break
        except Exception:
            pass
        time.sleep(1.0)

    if not healthy:
        for p in processes:
            p.terminate()
            p.wait()
        pytest.fail("Cluster failed to initialize.")

    # Sleep to allow health cache update
    time.sleep(6.0)

    yield

    for p in cluster_processes:
        if p.poll() is None:
            p.terminate()
            p.wait()

    for d in [SHARD0_A_DIR, SHARD0_B_DIR]:
        shutil.rmtree(d, ignore_errors=True)

@pytest.mark.asyncio
async def test_replica_replication_and_sync():
    coord_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_COORD, api_key="test-api-key-12345")
    
    # Direct access clients to check raw states
    headers = {"X-Internal-Token": "test-internal-token-67890"}
    client_a = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_S0_A}", headers=headers)
    client_b = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_S0_B}", headers=headers)
    
    try:
        # 1. Ingest memory through coordinator
        m1 = "Replication is a mechanism for data redundancy."
        id1 = await coord_db.remember(m1)
        assert id1.startswith("shard0-")
        
        # Give async SSE / replication task a split second
        await asyncio.sleep(0.5)

        # 2. Verify that it was successfully replicated to both replicas directly
        res_a = await client_a.post("/get", json={"memory_id": id1})
        res_b = await client_b.post("/get", json={"memory_id": id1})
        assert res_a.status_code == 200 and res_a.json().get("id") == id1
        assert res_b.status_code == 200 and res_b.json().get("id") == id1
        
        # 3. Simulate replica A outage (terminate process A)
        p_a = cluster_processes[0]
        p_a.terminate()
        p_a.wait()
        
        # Mark offline instantly in coordinator health loop or manually trigger polling update by sleeping
        await asyncio.sleep(6.0)

        # 4. Verify that reads still succeed via failover to replica B
        res_get = await coord_db.get(id1)
        assert res_get.id == id1
        
        # 5. Ingest new memory while replica A is offline
        m2 = "Consistency is maintained through synchronisation."
        id2 = await coord_db.remember(m2)
        assert id2.startswith("shard0-")
        
        await asyncio.sleep(0.5)

        # Verify that it exists on replica B
        res2_b = await client_b.post("/get", json={"memory_id": id2})
        assert res2_b.status_code == 200 and res2_b.json().get("id") == id2
        
        # 6. Recover replica A (restart process A)
        p_a_recovered = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S0_A)],
            env=cluster_envs[0], stdout=sys.stderr, stderr=sys.stderr
        )
        cluster_processes[0] = p_a_recovered
        
        # Wait for recovering replica A to start and coordinator polling loop to run sync (up to 15 seconds)
        healthy = False
        for _ in range(25):
            try:
                # Check status via coordinator stats to see if node becomes healthy and synced
                stats = await coord_db.stats()
                replica_status = stats.get("shards", {}).get(f"http://127.0.0.1:{PORT_S0_A}", {}).get("status")
                if replica_status == "healthy":
                    healthy = True
                    break
            except Exception:
                pass
            await asyncio.sleep(1.0)
            
        assert healthy, "Replica A failed to synchronize and transition back to healthy status."
        
        # 7. Verify that the memory written while offline is now synced on replica A
        res2_a = await client_a.post("/get", json={"memory_id": id2})
        assert res2_a.status_code == 200 and res2_a.json().get("id") == id2
        
    finally:
        await client_a.aclose()
        await client_b.aclose()


@pytest.mark.asyncio
async def test_write_consistency_levels():
    coord_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_COORD, api_key="test-api-key-12345")
    headers = {"X-Internal-Token": "test-internal-token-67890"}
    client_a = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_S0_A}", headers=headers)
    client_b = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_S0_B}", headers=headers)
    
    try:
        # Disable background sync to prevent automatic recovery sync from racing
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_COORD}", headers={"X-API-Key": "test-api-key-12345"}) as coord_client:
            await coord_client.post("/admin/toggle_sync", params={"enabled": "false"})

        # Write with consistency = "all" (should succeed since both nodes are online)
        id_all = await coord_db.remember("Durability is important.", consistency="all")
        assert id_all.startswith("shard0-")
        
        # Shut down replica A
        p_a = cluster_processes[0]
        if p_a.poll() is None:
            p_a.terminate()
            p_a.wait()
            
        await asyncio.sleep(1.0)
        
        # Write with consistency = "all" should now fail
        with pytest.raises(Exception):
            await coord_db.remember("This write should fail.", consistency="all")
            
        # Write with consistency = "one" should succeed even with one replica down
        id_one = await coord_db.remember("This write should succeed with ONE.", consistency="one")
        assert id_one.startswith("shard0-")
        
        # Recover replica A
        p_a_recovered = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S0_A)],
            env=cluster_envs[0], stdout=sys.stderr, stderr=sys.stderr
        )
        cluster_processes[0] = p_a_recovered
        
        # Wait for A to recover
        for _ in range(25):
            try:
                res = await client_a.get("/healthz")
                if res.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(1.0)
            
    finally:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_COORD}", headers={"X-API-Key": "test-api-key-12345"}) as coord_client:
            try:
                await coord_client.post("/admin/toggle_sync", params={"enabled": "true"})
            except Exception:
                pass
        await client_a.aclose()
        await client_b.aclose()


@pytest.mark.asyncio
async def test_read_repair_and_consistency():
    coord_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_COORD, api_key="test-api-key-12345")
    headers = {"X-Internal-Token": "test-internal-token-67890"}
    client_a = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_S0_A}", headers=headers)
    client_b = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_S0_B}", headers=headers)
    
    try:
        # Disable background sync to prevent automatic recovery sync from racing
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_COORD}", headers={"X-API-Key": "test-api-key-12345"}) as coord_client:
            await coord_client.post("/admin/toggle_sync", params={"enabled": "false"})

        # 1. Write a memory with ONE consistency
        m_id = await coord_db.remember("Original memory content.", consistency="one")
        await asyncio.sleep(0.5)
        
        # 2. Stop replica A process
        p_a = cluster_processes[0]
        if p_a.poll() is None:
            p_a.terminate()
            p_a.wait()
            
        # 3. Update memory (goes to replica B only, since A is down)
        await coord_db.update(m_id, text="Updated memory content.", consistency="one")
        
        # Verify B has updated content
        res_b = await client_b.post("/get", json={"memory_id": m_id})
        assert res_b.json().get("payload") == "Updated memory content."
        
        # 4. Recover replica A process
        p_a_recovered = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(PORT_S0_A)],
            env=cluster_envs[0], stdout=sys.stderr, stderr=sys.stderr
        )
        cluster_processes[0] = p_a_recovered
        
        # Wait for A to start up
        for _ in range(25):
            try:
                res = await client_a.get("/healthz")
                if res.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(1.0)
            
        # Manually verify that replica A still has the OLD memory content (since we haven't run cluster sync yet)
        res_a_old = await client_a.post("/get", json={"memory_id": m_id})
        assert res_a_old.json().get("payload") == "Original memory content."
        
        # 5. Perform a read via coordinator with QUORUM consistency (triggers Read Repair)
        point_mem = await coord_db.get(m_id, consistency="quorum")
        assert point_mem.text == "Updated memory content."
        
        # Give Read Repair background task a split second to execute
        await asyncio.sleep(1.5)
        
        # 6. Verify replica A has now been repaired and has the updated content!
        res_a_new = await client_a.post("/get", json={"memory_id": m_id})
        assert res_a_new.json().get("payload") == "Updated memory content."
        
    finally:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_COORD}", headers={"X-API-Key": "test-api-key-12345"}) as coord_client:
            try:
                await coord_client.post("/admin/toggle_sync", params={"enabled": "true"})
            except Exception:
                pass
        await client_a.aclose()
        await client_b.aclose()


@pytest.mark.asyncio
async def test_cluster_admin_reset():
    coord_db = AsyncRemoteEpochDB(host="127.0.0.1", port=PORT_COORD, api_key="test-api-key-12345")
    headers = {"X-Internal-Token": "test-internal-token-67890"}
    client_a = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_S0_A}", headers=headers)
    client_b = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_S0_B}", headers=headers)
    
    try:
        # Create a database reset call to the coordinator
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT_COORD}", headers={"X-API-Key": "test-api-key-12345"}) as coord_client:
            # 1. Ingest a tenant memory
            ingest_res = await coord_client.post(
                "/remember",
                json={"text": "Tenant memory content"},
                headers={"X-Tenant": "tenant123"}
            )
            assert ingest_res.status_code == 201
            m_id = ingest_res.json().get("id")
            
            # Verify B has it
            res_b = await client_b.post("/get", json={"memory_id": m_id}, headers={"X-Tenant": "tenant123"})
            assert res_b.status_code == 200 and res_b.json().get("id") == m_id
            
            # 2. Call admin reset on coordinator for this tenant
            reset_res = await coord_client.post(
                "/admin/reset",
                json={"tenant": "tenant123", "namespace": None}
            )
            assert reset_res.status_code == 200
            
            # 3. Verify it is deleted on both replicas directly
            res_a = await client_a.post("/get", json={"memory_id": m_id}, headers={"X-Tenant": "tenant123"})
            res_b_deleted = await client_b.post("/get", json={"memory_id": m_id}, headers={"X-Tenant": "tenant123"})
            assert res_a.json() == {}
            assert res_b_deleted.json() == {}
            
    finally:
        await client_a.aclose()
        await client_b.aclose()
