import os
import shutil
import time
import asyncio
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

import sys
# Configure environment variables BEFORE importing app
temp_dir = tempfile.mkdtemp(prefix="epochdb_multilevel_test_")
os.environ["NODE_MODE"] = "shard"
os.environ["INTERNAL_AUTH_TOKEN"] = "test-token-12345"
os.environ["STORAGE_DIR"] = temp_dir
os.environ["SHARD_EMBED_CONCURRENCY"] = "8"
os.environ["SHARD_EMBED_QUEUE_LIMIT"] = "100000"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.server import app, get_keystore, Permission


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_multilevel_hierarchical_scoping():
    """
    Tests multi-level hierarchical namespace scoping (e.g. tenant/dealer-group/dealer/store).
    Verifies:
    1. Ingestion into nested namespaces.
    2. Exact leaf namespace retrieval.
    3. Hierarchical prefix wildcard retrieval (group_WELLER/*).
    4. Tenant-wide wildcard retrieval (*).
    5. API key scope restriction on hierarchical paths.
    """
    tenant = "bmw-germany"
    admin_token = "test-token-12345"
    
    headers_store_a = {
        "X-Internal-Token": admin_token,
        "X-Tenant": tenant,
        "X-Namespace": "group_WELLER/store_34007",
    }
    headers_store_b = {
        "X-Internal-Token": admin_token,
        "X-Tenant": tenant,
        "X-Namespace": "group_WELLER/store_12345",
    }
    headers_store_c = {
        "X-Internal-Token": admin_token,
        "X-Tenant": tenant,
        "X-Namespace": "group_Graser/store_99999",
    }

    with TestClient(app) as client:
        # 1. Ingest memories into nested stores
        r1 = client.post(
            "/remember",
            json={"text": "Meister Krämmer at WELLER Store 34007 resolved brake noise efficiently."},
            headers=headers_store_a,
        )
        assert r1.status_code == 201
        id1 = r1.json()["id"]

        r2 = client.post(
            "/remember",
            json={"text": "Advisor Thomas at WELLER Store 12345 praised for clear battery diagnosis."},
            headers=headers_store_b,
        )
        assert r2.status_code == 201
        id2 = r2.json()["id"]

        r3 = client.post(
            "/remember",
            json={"text": "Schwemmer Bertl at Graser Store 99999 completed service inspection."},
            headers=headers_store_c,
        )
        assert r3.status_code == 201
        id3 = r3.json()["id"]

        # 2. Exact leaf lookup: Store A should only return Store A memories
        res_a = client.post(
            "/query",
            json={"query": "brake noise Meister Krämmer", "k": 5},
            headers=headers_store_a,
        )
        assert res_a.status_code == 200
        texts_a = [m.get("payload") or m.get("text") for m in res_a.json().get("results", [])]
        assert any("Meister Krämmer" in t for t in texts_a)
        assert not any("Schwemmer Bertl" in t for t in texts_a)

        # 3. Hierarchical prefix wildcard lookup (group_WELLER/*)
        # Should aggregate Store 34007 and Store 12345, but exclude Graser Store 99999!
        headers_weller_wildcard = {
            "X-Internal-Token": admin_token,
            "X-Tenant": tenant,
            "X-Namespace": "group_WELLER/*",
        }
        res_weller = client.post(
            "/query",
            json={"query": "WELLER", "k": 10},
            headers=headers_weller_wildcard,
        )
        assert res_weller.status_code == 200
        texts_weller = [m.get("payload") or m.get("text") for m in res_weller.json().get("results", [])]
        assert any("Meister Krämmer" in t for t in texts_weller)
        assert any("Advisor Thomas" in t for t in texts_weller)
        assert not any("Schwemmer Bertl" in t for t in texts_weller)

        # 4. Tenant-wide wildcard lookup (*)
        # Should find all stores across both WELLER and Graser
        headers_tenant_all = {
            "X-Internal-Token": admin_token,
            "X-Tenant": tenant,
            "X-Namespace": "*",
        }
        res_all = client.post(
            "/query",
            json={"query": "service diagnosis inspection", "k": 10},
            headers=headers_tenant_all,
        )
        assert res_all.status_code == 200
        texts_all = [m.get("payload") or m.get("text") for m in res_all.json().get("results", [])]
        assert any("Meister Krämmer" in t for t in texts_all)
        assert any("Schwemmer Bertl" in t for t in texts_all)

        # 5. Database partition listing should report nested hierarchy
        db_res = client.get("/databases", headers=headers_store_a)
        assert db_res.status_code == 200
        ns_list = [d["namespace"] for d in db_res.json().get("databases", []) if d.get("tenant") == tenant]
        assert "group_WELLER/store_34007" in ns_list
        assert "group_WELLER/store_12345" in ns_list
        assert "group_Graser/store_99999" in ns_list

        # 6. Scoped API Key restriction on hierarchical paths
        keystore = get_keystore()
        weller_key_id, weller_raw_key = keystore.create_key(
            permissions=[Permission.READ, Permission.WRITE],
            tenant=tenant,
            namespace="group_WELLER/*",
            description="Restricted to all WELLER stores"
        )

        # Authorized: accessing store under group_WELLER
        auth_resp = client.post(
            "/query",
            json={"query": "Krämmer", "k": 5},
            headers={"X-API-Key": weller_raw_key, "X-Tenant": tenant, "X-Namespace": "group_WELLER/store_34007"}
        )
        assert auth_resp.status_code == 200

        # Forbidden: attempting to access store under group_Graser
        forbidden_resp = client.post(
            "/query",
            json={"query": "Schwemmer", "k": 5},
            headers={"X-API-Key": weller_raw_key, "X-Tenant": tenant, "X-Namespace": "group_Graser/store_99999"}
        )
        assert forbidden_resp.status_code == 403


@pytest.mark.asyncio
async def test_simultaneous_writes_20k_resilience():
    """
    Stress test verifying EpochDB handles 20,000 simultaneous writes without hanging or crashing.
    Simulates high-concurrency write burst from thousands of simultaneous users.
    """
    from epochdb.engine import EpochDB

    storage_dir = os.path.join(temp_dir, "resilience_20k")
    dim = 64
    base_emb = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

    engine = EpochDB(
        storage_dir=storage_dir,
        dim=dim,
        model=None,  # Direct embedding mode for high-throughput concurrency benchmark
        wal_sync_interval=0.05,
    )

    TOTAL_WRITES = 20_000
    BATCH_CHUNK = 1000

    items = [
        {
            "payload": f"Customer feedback #{i}: Express oil and brake service completed by technician T-{i % 100} at station {i % 10}.",
            "embedding": base_emb,
            "triples": [(f"Technician_T{i % 100}", "works_at", f"Station_{i % 10}")],
            "metadata": {
                "customer_idx": i,
                "technician_id": f"T-{i % 100}",
                "station": i % 10,
                "sentiment": "positive" if i % 2 == 0 else "neutral",
            },
            "atom_id": f"atom-20k-{i}",
        }
        for i in range(TOTAL_WRITES)
    ]

    start_time = time.time()

    # Dispatch 20 concurrent batches of 1,000 writes across threads simulating simultaneous write bursts
    loop = asyncio.get_running_loop()
    tasks = []
    for chunk_start in range(0, TOTAL_WRITES, BATCH_CHUNK):
        chunk = items[chunk_start : chunk_start + BATCH_CHUNK]
        tasks.append(loop.run_in_executor(None, engine.add_memory_batch, chunk))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start_time

    total_written = 0
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            pytest.fail(f"Task chunk {idx} failed with error: {res}")
        assert isinstance(res, list)
        total_written += len(res)

    assert total_written == TOTAL_WRITES, f"Expected {TOTAL_WRITES} writes, got {total_written}"
    assert engine.get_total_atoms() == TOTAL_WRITES

    throughput = TOTAL_WRITES / max(elapsed, 0.001)
    print(f"\n[20K WRITES RESILIENCE SUCCESS] Ingested {TOTAL_WRITES:,} writes in {elapsed:.2f}s ({throughput:,.0f} writes/sec) with 0 errors and 0 crashes!")

    # Verify retrieval from the ingested 20,000 memories
    associations = engine.kg_manager.get_associations("Technician_T42")
    assert len(associations) > 0
    atoms = engine.recall_by_entity("Technician_T42")
    assert len(atoms) > 0
    assert any("Technician_T42" in t[0] for t in atoms[0].triples)
