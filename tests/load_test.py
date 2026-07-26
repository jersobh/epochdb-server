#!/usr/bin/env python3
import asyncio
import time
import argparse
import random
import statistics
import json
import sys
import os
import shutil
import tempfile
import subprocess
from typing import List, Dict, Any, Tuple
import httpx

# -------------------------------------------------------------------------
# Dynamic Fact Generation Pool
# -------------------------------------------------------------------------
SUBJECTS = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", 
    "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Pete", "Quinn", 
    "Rose", "Sam", "Toby", "Alexander", "Sophia", "Zoe", "William", "Lucas"
]
PREDICATES = [
    "works_at", "lives_in", "loves", "hates", "owns", "created", "visited", 
    "studied", "teaches", "designed", "discovered", "wrote", "built", "repaired"
]
OBJECTS = [
    "Google", "London", "Paris", "Python", "Rust", "SQLite", "HNSW", "EpochDB", 
    "GitHub", "Docker", "Machine Learning", "Artificial Intelligence", "TypeScript",
    "Tokyo", "Berlin", "San Francisco", "React", "FastAPI", "Kubernetes", "Redis"
]

def generate_random_fact() -> Tuple[str, Dict[str, Any], str, str]:
    sub = random.choice(SUBJECTS)
    pred = random.choice(PREDICATES)
    obj = random.choice(OBJECTS)
    text = f"{sub} {pred.replace('_', ' ')} {obj}."
    metadata = {"triples": [(sub, pred, obj)], "type": "synthetic_load_test"}
    return text, metadata, sub, obj

# -------------------------------------------------------------------------
# Metrics Tracking
# -------------------------------------------------------------------------
class MetricTracker:
    def __init__(self):
        self.success_latencies: Dict[str, List[float]] = {}
        self.failures: Dict[str, int] = {}
        self.lock = asyncio.Lock()

    def register_endpoint(self, name: str):
        if name not in self.success_latencies:
            self.success_latencies[name] = []
            self.failures[name] = 0

    async def record_success(self, endpoint: str, latency: float):
        async with self.lock:
            self.success_latencies[endpoint].append(latency)

    async def record_failure(self, endpoint: str):
        async with self.lock:
            self.failures[endpoint] += 1

def calculate_percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(round(pct * (len(sorted_data) - 1) / 100.0))
    return sorted_data[idx]

# -------------------------------------------------------------------------
# Cluster Orchestration (For self-contained local runs)
# -------------------------------------------------------------------------
class LocalClusterManager:
    def __init__(self):
        self.processes = []
        self.temp_dirs = []
        self.api_key = "test-api-key-12345"
        self.internal_token = "test-internal-token-67890"

    def start_cluster(self) -> str:
        print("=== Launching Local Clustered Test Environment ===")
        # Port assignments
        port_coord = 28080
        ports_shards = [28081, 28082, 28083]

        for i, port in enumerate(ports_shards):
            temp_dir = tempfile.mkdtemp(prefix=f"epochdb_load_shard{i}_")
            self.temp_dirs.append(temp_dir)
            
            env = os.environ.copy()
            env["NODE_MODE"] = "shard"
            env["STORAGE_DIR"] = temp_dir
            env["INTERNAL_AUTH_TOKEN"] = self.internal_token
            
            print(f"Starting Shard {i} on port {port} (Data: {temp_dir})...")
            p = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(port)],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            self.processes.append(p)

        # Start coordinator
        env_coord = os.environ.copy()
        env_coord["NODE_MODE"] = "coordinator"
        env_coord["SHARD_NODES"] = ",".join([f"http://127.0.0.1:{p}" for p in ports_shards])
        env_coord["API_KEY"] = self.api_key
        env_coord["INTERNAL_AUTH_TOKEN"] = self.internal_token

        print(f"Starting Coordinator Gateway on port {port_coord}...")
        p_coord = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.server:app", "--host", "127.0.0.1", "--port", str(port_coord)],
            env=env_coord, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self.processes.append(p_coord)

        # Wait for nodes to pass health checks (first boot may download embedder weights).
        time.sleep(2.0)
        start_wait = time.time()
        nodes_healthy = False
        last_err = None
        while time.time() - start_wait < 90:
            try:
                healthy_count = 0
                for port in [port_coord] + ports_shards:
                    resp = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
                    if resp.status_code == 200:
                        healthy_count += 1
                if healthy_count == 4:
                    nodes_healthy = True
                    break
            except Exception as e:
                last_err = e
            time.sleep(1.0)

        if not nodes_healthy:
            # Surface process exits to debug CI flakes.
            for i, p in enumerate(self.processes):
                code = p.poll()
                if code is not None:
                    print(f"Process[{i}] exited early with code={code}")
            self.stop_cluster()
            raise RuntimeError(
                f"Local cluster failed to initialize healthy status in time. last_err={last_err!r}"
            )

        # Let the coordinator build its internal health cache
        print("Cluster is up. Waiting 5s for topology sync...")
        time.sleep(5.0)
        return f"http://127.0.0.1:{port_coord}"

    def stop_cluster(self):
        print("=== Shutting Down Local Clustered Test Environment ===")
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
                p.wait()
        for d in self.temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        print("Cluster cleaned up cleanly.")

# -------------------------------------------------------------------------
# Load Generator Worker
# -------------------------------------------------------------------------
async def load_generator_worker(
    worker_id: int,
    target_url: str,
    headers: Dict[str, str],
    tracker: MetricTracker,
    seeded_memories: List[Dict[str, Any]],
    memories_lock: asyncio.Lock,
    operations: List[str],
    weights: List[int],
    duration: float,
    start_time: float
):
    # Setup client with connection pooling properties
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    async with httpx.AsyncClient(headers=headers, timeout=15.0, limits=limits) as client:
        while time.perf_counter() - start_time < duration:
            op = random.choices(operations, weights=weights, k=1)[0]
            
            # 1. GET
            if op == "get":
                async with memories_lock:
                    mem = random.choice(seeded_memories) if seeded_memories else None
                if not mem:
                    continue
                
                t_start = time.perf_counter()
                try:
                    resp = await client.post(f"{target_url}/get", json={"memory_id": mem["id"]})
                    latency = time.perf_counter() - t_start
                    if resp.status_code == 200:
                        await tracker.record_success("POST /get", latency)
                    else:
                        await tracker.record_failure("POST /get")
                except Exception:
                    await tracker.record_failure("POST /get")

            # 2. QUERY
            elif op == "query":
                async with memories_lock:
                    mem = random.choice(seeded_memories) if seeded_memories else None
                query_str = f"Who studied {mem['object']}?" if mem else "Search query string"
                
                t_start = time.perf_counter()
                try:
                    resp = await client.post(f"{target_url}/query", json={"query": query_str, "k": 3})
                    latency = time.perf_counter() - t_start
                    if resp.status_code == 200:
                        await tracker.record_success("POST /query", latency)
                    else:
                        await tracker.record_failure("POST /query")
                except Exception:
                    await tracker.record_failure("POST /query")

            # 3. REMEMBER (Write)
            elif op == "remember":
                fact, meta, sub, obj = generate_random_fact()
                t_start = time.perf_counter()
                try:
                    resp = await client.post(f"{target_url}/remember", json={"text": fact, "metadata": meta})
                    latency = time.perf_counter() - t_start
                    if resp.status_code == 201:
                        await tracker.record_success("POST /remember", latency)
                        resp_data = resp.json()
                        new_id = resp_data.get("id")
                        if new_id:
                            async with memories_lock:
                                seeded_memories.append({"id": new_id, "subject": sub, "object": obj})
                    else:
                        await tracker.record_failure("POST /remember")
                except Exception:
                    await tracker.record_failure("POST /remember")

            # 4. UPDATE
            elif op == "update":
                async with memories_lock:
                    mem = random.choice(seeded_memories) if seeded_memories else None
                if not mem:
                    continue
                
                new_fact = f"{mem['subject']} updated study target."
                t_start = time.perf_counter()
                try:
                    resp = await client.post(f"{target_url}/update", json={"memory_id": mem["id"], "text": new_fact})
                    latency = time.perf_counter() - t_start
                    if resp.status_code == 200:
                        await tracker.record_success("POST /update", latency)
                    else:
                        await tracker.record_failure("POST /update")
                except Exception:
                    await tracker.record_failure("POST /update")

            # 5. DELETE
            elif op == "delete":
                async with memories_lock:
                    if len(seeded_memories) > 10:  # Prevent completely draining our read cache
                        mem = seeded_memories.pop(random.randint(0, len(seeded_memories) - 1))
                    else:
                        mem = None
                if not mem:
                    continue
                
                t_start = time.perf_counter()
                try:
                    resp = await client.post(f"{target_url}/delete", json={"memory_id": mem["id"], "hard": False})
                    latency = time.perf_counter() - t_start
                    if resp.status_code == 200:
                        await tracker.record_success("POST /delete", latency)
                    else:
                        await tracker.record_failure("POST /delete")
                        # put back
                        async with memories_lock:
                            seeded_memories.append(mem)
                except Exception:
                    await tracker.record_failure("POST /delete")
                    async with memories_lock:
                        seeded_memories.append(mem)

            # 6. ENTITY GRAPH
            elif op == "entity_graph":
                async with memories_lock:
                    mem = random.choice(seeded_memories) if seeded_memories else None
                entity = mem["subject"] if mem else "Zoe"
                
                t_start = time.perf_counter()
                try:
                    resp = await client.get(f"{target_url}/entity_graph?entity_id={entity}&depth=1")
                    latency = time.perf_counter() - t_start
                    if resp.status_code == 200:
                        await tracker.record_success("GET /entity_graph", latency)
                    else:
                        await tracker.record_failure("GET /entity_graph")
                except Exception:
                    await tracker.record_failure("GET /entity_graph")

            # 7. GET TIMELINE
            elif op == "get_timeline":
                async with memories_lock:
                    mem = random.choice(seeded_memories) if seeded_memories else None
                entity = mem["subject"] if mem else "Zoe"
                
                t_start = time.perf_counter()
                try:
                    resp = await client.post(f"{target_url}/get_timeline", json={"entity_id": entity})
                    latency = time.perf_counter() - t_start
                    if resp.status_code == 200:
                        await tracker.record_success("POST /get_timeline", latency)
                    else:
                        await tracker.record_failure("POST /get_timeline")
                except Exception:
                    await tracker.record_failure("POST /get_timeline")

            # 8. STATS
            elif op == "stats":
                t_start = time.perf_counter()
                try:
                    resp = await client.get(f"{target_url}/stats")
                    latency = time.perf_counter() - t_start
                    if resp.status_code == 200:
                        await tracker.record_success("GET /stats", latency)
                    else:
                        await tracker.record_failure("GET /stats")
                except Exception:
                    await tracker.record_failure("GET /stats")

            # Yield to event loop to allow concurrent task switching
            await asyncio.sleep(0.0001)

# -------------------------------------------------------------------------
# Seed Data Setup
# -------------------------------------------------------------------------
async def seed_database(target_url: str, headers: Dict[str, str], count: int) -> List[Dict[str, Any]]:
    print(f"Seeding database with {count} initial memory items...")
    seeded_items = []
    
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        for i in range(count):
            fact, meta, sub, obj = generate_random_fact()
            try:
                resp = await client.post(f"{target_url}/remember", json={"text": fact, "metadata": meta})
                if resp.status_code == 201:
                    new_id = resp.json().get("id")
                    if new_id:
                        seeded_items.append({"id": new_id, "subject": sub, "object": obj})
                else:
                    print(f"  [Warning] Ingestion seed {i} failed: HTTP {resp.status_code}")
            except Exception as e:
                print(f"  [Warning] Ingestion seed {i} raised error: {e}")
                
    print(f"Successfully seeded {len(seeded_items)} records.")
    return seeded_items

# -------------------------------------------------------------------------
# CLI & Runner Orchestrator
# -------------------------------------------------------------------------
async def main_async():
    parser = argparse.ArgumentParser(description="EpochDB high-performance asynchronous load testing suite.")
    parser.add_argument("--target", "-t", default="http://localhost:8080", help="Target coordinator/shard base URL.")
    parser.add_argument("--api-key", "-k", default="test-api-key-12345", help="Client authentication gateway key.")
    parser.add_argument("--internal-token", "-i", default="test-internal-token-67890", help="Internal shard auth token.")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Number of concurrent virtual user client tasks.")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Duration of the load test in seconds.")
    parser.add_argument("--seed-count", "-s", type=int, default=50, help="Number of database items to seed beforehand.")
    parser.add_argument("--shard-mode", action="store_true", help="Targets a direct shard storage node directly (uses internal token instead of api key).")
    parser.add_argument("--run-local-cluster", action="store_true", help="Spin up a temporary local 3-shard cluster automatically for execution.")
    parser.add_argument("--ratio", type=str, default="10:35:35:10:2:3:3:2", help="Weight ratios for endpoints (remember:get:query:update:delete:entity_graph:get_timeline:stats).")
    parser.add_argument("--output", "-o", default=None, help="Save final metrics summary reports to JSON file.")
    
    args = parser.parse_args()

    # Parse ratios
    r_parts = [int(x) for x in args.ratio.split(":")]
    if len(r_parts) != 8:
        print("Error: --ratio must contain exactly 8 parts separated by ':'")
        sys.exit(1)

    operations = ["remember", "get", "query", "update", "delete", "entity_graph", "get_timeline", "stats"]
    weights = r_parts

    cluster_manager = None
    target_url = args.target
    
    # Configure headers
    headers = {}
    if args.shard_mode:
        headers["X-Internal-Token"] = args.internal_token
    else:
        headers["X-API-Key"] = args.api_key

    # Spin up cluster if requested
    if args.run_local_cluster:
        cluster_manager = LocalClusterManager()
        try:
            target_url = cluster_manager.start_cluster()
            headers = {"X-API-Key": cluster_manager.api_key}
        except Exception as e:
            print(f"Error starting local cluster: {e}")
            sys.exit(1)

    tracker = MetricTracker()
    for op in operations:
        tracker.register_endpoint(f"POST /{op}" if op not in ["entity_graph", "stats"] else f"GET /{op}")

    # Seed Database
    seeded_memories = await seed_database(target_url, headers, args.seed_count)
    if not seeded_memories:
        print("Database could not be seeded. Exiting load test.")
        if cluster_manager:
            cluster_manager.stop_cluster()
        sys.exit(1)

    memories_lock = asyncio.Lock()

    print("\n======================================================================")
    print("RUNNING DISTRIBUTED LOAD TEST SUITE")
    print(f"  Target URL:   {target_url}")
    print(f"  Concurrency:  {args.concurrency} virtual client tasks")
    print(f"  Duration:     {args.duration} seconds")
    print(f"  Ratio Weights: {dict(zip(operations, weights))}")
    print("======================================================================\n")

    # Start the test
    start_time = time.perf_counter()
    tasks = []
    for w_id in range(args.concurrency):
        task = asyncio.create_task(
            load_generator_worker(
                worker_id=w_id,
                target_url=target_url,
                headers=headers,
                tracker=tracker,
                seeded_memories=seeded_memories,
                memories_lock=memories_lock,
                operations=operations,
                weights=weights,
                duration=args.duration,
                start_time=start_time
            )
        )
        tasks.append(task)

    # Let workers run for the duration
    await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    actual_duration = end_time - start_time

    # Stop local cluster if we started it
    if cluster_manager:
        cluster_manager.stop_cluster()

    # Process and Display results
    print("\n======================================================================")
    print("LOAD TEST COMPLETE - METRICS OVERVIEW")
    print("======================================================================\n")

    report_data = {
        "target": target_url,
        "concurrency": args.concurrency,
        "duration_seconds": actual_duration,
        "endpoints": {}
    }

    # Print Table Header
    print(f"{'Endpoint':<20} | {'Count':<7} | {'Success':<7} | {'Failures':<8} | {'RPS':<6} | {'Avg (ms)':<8} | {'p50 (ms)':<8} | {'p90 (ms)':<8} | {'p95 (ms)':<8} | {'p99 (ms)':<8}")
    print("-" * 112)

    total_reqs = 0
    total_failures = 0
    all_latencies = []

    for op in operations:
        name = f"POST /{op}" if op not in ["entity_graph", "stats"] else f"GET /{op}"
        latencies = tracker.success_latencies.get(name, [])
        fail_count = tracker.failures.get(name, 0)
        succ_count = len(latencies)
        op_reqs = succ_count + fail_count
        
        total_reqs += op_reqs
        total_failures += fail_count
        all_latencies.extend(latencies)

        op_rps = op_reqs / actual_duration if actual_duration > 0 else 0
        avg_lat = statistics.mean(latencies) * 1000.0 if latencies else 0.0
        p50 = calculate_percentile(latencies, 50.0) * 1000.0 if latencies else 0.0
        p90 = calculate_percentile(latencies, 90.0) * 1000.0 if latencies else 0.0
        p95 = calculate_percentile(latencies, 95.0) * 1000.0 if latencies else 0.0
        p99 = calculate_percentile(latencies, 99.0) * 1000.0 if latencies else 0.0

        print(f"{name:<20} | {op_reqs:<7} | {succ_count:<7} | {fail_count:<8} | {op_rps:<6.1f} | {avg_lat:<8.1f} | {p50:<8.1f} | {p90:<8.1f} | {p95:<8.1f} | {p99:<8.1f}")
        
        report_data["endpoints"][name] = {
            "total_requests": op_reqs,
            "success_count": succ_count,
            "failure_count": fail_count,
            "rps": op_rps,
            "latency_avg_ms": avg_lat,
            "latency_p50_ms": p50,
            "latency_p90_ms": p90,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99
        }

    print("-" * 112)
    overall_rps = total_reqs / actual_duration if actual_duration > 0 else 0
    overall_avg = statistics.mean(all_latencies) * 1000.0 if all_latencies else 0.0
    overall_p50 = calculate_percentile(all_latencies, 50.0) * 1000.0 if all_latencies else 0.0
    overall_p90 = calculate_percentile(all_latencies, 90.0) * 1000.0 if all_latencies else 0.0
    overall_p95 = calculate_percentile(all_latencies, 95.0) * 1000.0 if all_latencies else 0.0
    overall_p99 = calculate_percentile(all_latencies, 99.0) * 1000.0 if all_latencies else 0.0
    
    print(f"{'OVERALL':<20} | {total_reqs:<7} | {total_reqs - total_failures:<7} | {total_failures:<8} | {overall_rps:<6.1f} | {overall_avg:<8.1f} | {overall_p50:<8.1f} | {overall_p90:<8.1f} | {overall_p95:<8.1f} | {overall_p99:<8.1f}")
    
    report_data["overall"] = {
        "total_requests": total_reqs,
        "success_count": total_reqs - total_failures,
        "failure_count": total_failures,
        "rps": overall_rps,
        "latency_avg_ms": overall_avg,
        "latency_p50_ms": overall_p50,
        "latency_p90_ms": overall_p90,
        "latency_p95_ms": overall_p95,
        "latency_p99_ms": overall_p99
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nReport written to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main_async())


def test_load_test_integration():
    """
    pytest integration test hook. Runs a quick 1-second local cluster run
    to verify that the load testing script is functional.
    """
    import subprocess
    import sys
    
    # Run a short 1-second dry-run of this script in a subprocess
    result = subprocess.run([
        sys.executable, __file__,
        "--run-local-cluster",
        "--concurrency", "1",
        "--duration", "1",
        "--seed-count", "1"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Load test execution failed:\n{result.stderr}\n{result.stdout}"
    assert "LOAD TEST COMPLETE" in result.stdout

