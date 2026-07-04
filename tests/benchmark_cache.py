import os
import sys
import time
import asyncio
import httpx
import tempfile
import shutil
import statistics

# Setup environment variables BEFORE importing app so they are read during module load
temp_dir = tempfile.mkdtemp(prefix="epochdb_benchmark_")
os.environ["NODE_MODE"] = "shard"
os.environ["INTERNAL_AUTH_TOKEN"] = "benchmark-token-12345"
os.environ["STORAGE_DIR"] = temp_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.server import app, CLUSTER_STATE_VERSIONS, LOCAL_READ_CACHE

async def run_benchmark():
    headers = {"X-Internal-Token": "benchmark-token-12345"}
    num_requests = 100
    
    print("======================================================================")
    print("EPOCHDB CACHE LAYER LATENCY & THROUGHPUT BENCHMARK")
    print("======================================================================")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Ingest test memories
        print("Ingesting test data...")
        mem_ids = []
        for i in range(10):
            res = await client.post(
                "/remember", 
                json={"text": f"Benchmark memory record number {i} for performance evaluation."}, 
                headers=headers
            )
            mem_ids.append(res.json()["id"])
            
        print(f"Ingested {len(mem_ids)} memories successfully.\n")

        # --- BENCHMARK 1: GET (Cold Cache / Miss) ---
        get_miss_times = []
        for mem_id in mem_ids * (num_requests // len(mem_ids)):
            # Force invalidate cache before each get to guarantee a cache miss/database read
            LOCAL_READ_CACHE.clear()
            start = time.perf_counter()
            resp = await client.post("/get", json={"memory_id": mem_id}, headers=headers)
            end = time.perf_counter()
            assert resp.status_code == 200
            get_miss_times.append(end - start)

        # --- BENCHMARK 2: GET (Warm Cache / Coordinator Hit) ---
        get_hit_times = []
        for mem_id in mem_ids * (num_requests // len(mem_ids)):
            start = time.perf_counter()
            resp = await client.post("/get", json={"memory_id": mem_id}, headers=headers)
            end = time.perf_counter()
            assert resp.status_code == 200
            get_hit_times.append(end - start)

        # --- BENCHMARK 3: GET (HTTP 304 Validation / ETag Match) ---
        get_304_times = []
        # First, fetch once to get the ETags
        etags = {}
        for mem_id in mem_ids:
            resp = await client.post("/get", json={"memory_id": mem_id}, headers=headers)
            etags[mem_id] = resp.headers.get("etag")

        for mem_id in mem_ids * (num_requests // len(mem_ids)):
            headers_with_etag = {**headers, "If-None-Match": etags[mem_id]}
            start = time.perf_counter()
            resp = await client.post("/get", json={"memory_id": mem_id}, headers=headers_with_etag)
            end = time.perf_counter()
            assert resp.status_code == 304
            get_304_times.append(end - start)

        # --- BENCHMARK 4: QUERY (Cold Cache / Miss) ---
        query_payload = {"query": "performance evaluation", "k": 3}
        query_miss_times = []
        for _ in range(num_requests):
            LOCAL_READ_CACHE.clear()
            start = time.perf_counter()
            resp = await client.post("/query", json=query_payload, headers=headers)
            end = time.perf_counter()
            assert resp.status_code == 200
            query_miss_times.append(end - start)

        # --- BENCHMARK 5: QUERY (Warm Cache / Coordinator Hit) ---
        query_hit_times = []
        for _ in range(num_requests):
            start = time.perf_counter()
            resp = await client.post("/query", json=query_payload, headers=headers)
            end = time.perf_counter()
            assert resp.status_code == 200
            query_hit_times.append(end - start)

        # --- BENCHMARK 6: QUERY (HTTP 304 Validation / ETag Match) ---
        resp = await client.post("/query", json=query_payload, headers=headers)
        etag_query = resp.headers.get("etag")
        
        query_304_times = []
        for _ in range(num_requests):
            headers_with_etag = {**headers, "If-None-Match": etag_query}
            start = time.perf_counter()
            resp = await client.post("/query", json=query_payload, headers=headers_with_etag)
            end = time.perf_counter()
            assert resp.status_code == 304
            query_304_times.append(end - start)

    # Clean up storage
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Print results
    def format_results(times):
        total_time = sum(times)
        mean_time_ms = statistics.mean(times) * 1000.0
        median_time_ms = statistics.median(times) * 1000.0
        qps = len(times) / total_time
        return mean_time_ms, median_time_ms, qps

    get_miss_mean, get_miss_med, get_miss_qps = format_results(get_miss_times)
    get_hit_mean, get_hit_med, get_hit_qps = format_results(get_hit_times)
    get_304_mean, get_304_med, get_304_qps = format_results(get_304_times)

    query_miss_mean, query_miss_med, query_miss_qps = format_results(query_miss_times)
    query_hit_mean, query_hit_med, query_hit_qps = format_results(query_hit_times)
    query_304_mean, query_304_med, query_304_qps = format_results(query_304_times)

    print("\n### BENCHMARK SUMMARY TABLE\n")
    print("| Operation | Cache Type | Mean Latency (ms) | Median Latency (ms) | Throughput (QPS) | Speedup Factor |")
    print("| --- | --- | --- | --- | --- | --- |")
    print(f"| `POST /get` | Cache Miss (Cold) | {get_miss_mean:.3f} ms | {get_miss_med:.3f} ms | {get_miss_qps:.1f} QPS | 1.0x (Baseline) |")
    print(f"| `POST /get` | In-Memory Hit (Warm) | {get_hit_mean:.3f} ms | {get_hit_med:.3f} ms | {get_hit_qps:.1f} QPS | {get_miss_mean/get_hit_mean:.1f}x |")
    print(f"| `POST /get` | HTTP 304 Validation | {get_304_mean:.3f} ms | {get_304_med:.3f} ms | {get_304_qps:.1f} QPS | {get_miss_mean/get_304_mean:.1f}x |")
    print(f"| `POST /query` | Cache Miss (Cold) | {query_miss_mean:.3f} ms | {query_miss_med:.3f} ms | {query_miss_qps:.1f} QPS | 1.0x (Baseline) |")
    print(f"| `POST /query` | In-Memory Hit (Warm) | {query_hit_mean:.3f} ms | {query_hit_med:.3f} ms | {query_hit_qps:.1f} QPS | {query_miss_mean/query_hit_mean:.1f}x |")
    print(f"| `POST /query` | HTTP 304 Validation | {query_304_mean:.3f} ms | {query_304_med:.3f} ms | {query_304_qps:.1f} QPS | {query_miss_mean/query_304_mean:.1f}x |")
    print("\n======================================================================")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
