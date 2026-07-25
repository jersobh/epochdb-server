# Changelog

All notable changes to the EpochDB Distributed Server project will be documented in this file.

## [0.9.7] - 2026-07-25
### Changed
- **Dependency bump**: Updated `epochdb` dependency requirement to `>=1.8.0`.
- **Pairwise Entity Graph Extraction**: Updated `/remember` memory fallback entity extraction to generate pairwise co-occurrence triples `(entity1, "co_occurs_with", entity2)` between distinct entities.
- **Visualization Scope Controls & Input Autofill Fix**:
  - Added Tenant and Namespace input fields to the Subgraph Filter panel in `visualize.html`, sending `X-Tenant` / `X-Namespace` headers on API requests.
  - Added `name` attributes and `autocomplete="off"` to inputs in `visualize.html` to prevent browser autofill and auto-filtering on page refresh.

## [0.9.6] - 2026-07-23
### Added
- **DuckDB Analytics SQL Endpoint**: Added `POST /v1/analytics/query` REST API route allowing clients to execute DuckDB SQL queries over historical memory archives.
- **Dependency bump**: Updated `epochdb` dependency requirement to `>=1.7.0` and added `duckdb (>=1.5.5,<2.0.0)`.

## [0.9.5]
### Changed
- **Dependency bump**: Updated `epochdb` dependency requirement to `>=1.6.2` to pull in fixes for query candidate overwriting and compaction deduplication.

## [0.9.4]
### Added
- **Fix visualization updates**: Fixed the issue where the visualization updates were not working as expected. The new state is published to the SSE stream whenever a value changes.

## [0.9.3] - 2026-07-08
### Fixed
- **Read Repair Test Race Condition**: Added `/admin/toggle_sync` coordinator endpoint to temporarily disable background health synchronization in integration tests, preventing automated recovery loops from racing with test verification assertions.

## [0.9.2] - 2026-07-08
### Fixed
- **Admin Reset Deletion Order**: Fixed order of operations in `/admin/reset` where storage directory deletion was executed after re-initialization of the default database instance, causing subsequent calls to fail.
- **databases Empty Reset List**: Updated `/admin/databases` list operation to scan active in-memory `db_pool` keys, guaranteeing empty or newly reset database partitions are detected and synchronized during recovery loops.
- **remember Endpoint memory_type Parameter**: Cleaned up the `/remember` endpoint to directly delegate `memory_type` to the underlying engine's `remember` API, removing manual retrieval and property-setting workarounds.

## [0.9.1] - 2026-07-08
### Added
- **Multi-Cloud Deployment Blueprints**:
  - Added Terraform templates to provision managed Kubernetes clusters on AWS (EKS), GCP (GKE), and Azure (AKS).
  - Added Kubernetes manifests including a dedicated Namespace, ConfigMap/Secret credentials, StatefulSet with a Headless Service for sharded storage nodes, and a LoadBalancer-exposed Coordinator Deployment.
  - Added a comprehensive step-by-step Multi-Cloud Deployment Guide at `deploy/README.md`.

## [0.9.0] - 2026-07-08
### Added
- **Extended Replica Set Replication**:
  - Implemented configurable Write Consistency Levels (`ONE`, `QUORUM`, `ALL`) to guarantee durability across replica sets.
  - Implemented Point Read (`/get`) multi-replica queries with active background **Read Repair** to automatically heal stale or missing replica states.
  - Implemented cluster-wide partition resets (`/admin/reset`) broadcasting administrative actions across all shards and replicas in the cluster.
  - Added new integration tests validating consistency levels and Read Repair logic.

## [0.8.0] - 2026-07-04
### Added
- **Distributed Concurrency Load Testing Suite**: Added `tests/load_test.py` and `tests/locustfile.py` to evaluate cluster performance. The custom async script supports auto-seeding, configurable ratios, detailed statistics table (averages, percentiles), and a self-contained local cluster deployment option.
- **pytest Integration Hook**: Appended integration test case to `load_test.py` to support scanning and dry-runs under standard test frameworks (`pytest tests/load_test.py`).
- **Dynamic Embedding Provider Config**: Exposes `EMBEDDING_MODEL` and `EMBEDDING_DIM` configuration via environment variables in `src/server.py` and propagates them to downstream database shards, enabling on-the-fly cloud providers (OpenAI, Gemini, etc.).
- **Docker Compose Integration**: Configured `docker-compose.yml` to automatically inject host embedding variables to the database shards.
- **Config Templates**: Added `.env.example` to provide dynamic configuration environment templates for both the core library and server.

## [0.7.0] - 2026-07-04
### Added
- **Zero-Dependency Write-Invalidated Cache**: Implemented an in-memory cache layer directly on the coordinator gateway.
  - Support for client-side HTTP cache validation (`If-None-Match` / `304 Not Modified`) using versioned ETags, yielding a **2.3x speedup** on direct memory lookups.
  - Local query caching for semantic queries, reducing query latency by **11.5x** (from 56ms to <5ms).
  - Context-aware cache namespaces isolated by tenant and namespace.
  - Safe write invalidation: mutations (`POST /remember`, `POST /update`, `POST /delete`, `POST /compact`) increment context state versions and flush the read cache, ensuring zero stale data is returned.
- **Cache Benchmark**: Added a latency and throughput performance runner at `tests/benchmark_cache.py`.

## [0.6.0] - 2026-07-04
### Added
- **Server-Sent Events (SSE)**: Added `/stream` endpoint to stream database mutation notifications to clients.
- **Real-Time Visualizer Updates**: Integrated EventSource SSE connection in `visualize.html` to instantly update the graph/stats on writes/deletes, eliminating 5-second HTTP polling traffic.

## [0.5.0] - 2026-07-04
### Added
- **Adaptive Querying Gateway**: Exposed `/adaptive_query` REST endpoint on coordinator gateway and shards, enabling LLM-orchestrated routing.
- **Context Window Propagation**: Exposed `context_window` in `/query` and `/adaptive_query` payloads, automatically propagating context parameters downstream to storage shards.

## [0.3.1] - 2026-06-28

### Added
- **Visualizer HTML Modularization**: Separated client-side visualizer code into a dedicated, clean `visualize.html` template file.
- **Brand Logo & Custom Sliders**: Added `/logo.png` route to serve the brand logo, replaced the placeholder infinity icon, and customized range sliders with premium orange CSS rules matching the dashboard's design system.
- **Robust Client Query Token Integration**: Implemented automatic query string token parameter extraction, local storage persistence, and page history cleaning, ensuring persistent session state and data recovery across server restarts.

### Changed
- **Changelogs & Dependencies**: Updated dependency on `epochdb` to `==1.3.1` to incorporate database timeline recovery and AsyncEpochDB argument bug fixes.

## [0.3.0] - 2026-06-27

### Added
- **API Key Security & Internal Tokens**:
  - Implemented token-based authentication on the coordinator gateway (`X-API-Key`) and shard nodes (`X-Internal-Token`).
  - The coordinator automatically propagates the `X-Internal-Token` header to downstream shards.
  - Client SDK (`AsyncRemoteEpochDB`) now supports optional `api_key` initialization.
- **Production ASGI Process Management**:
  - Replaced single-worker Uvicorn invocation with **Gunicorn** process manager running `uvicorn.workers.UvicornWorker`.
  - Configured worker timeout limit to 120s to ensure embedding models warm up cleanly without getting SIGKILL from Gunicorn.
- **Stale Lock Cleanup**:
  - Pre-purges `/data/.lock` file prior to booting Gunicorn inside the container, preventing database lock crashes when Docker volumes are mounted persistently.
- **Urllib-Based Health Probes**:
  - Added a `/healthz` endpoint on shards (readiness indicator) and the coordinator (which polls shards' health).
  - Integrated Docker `healthcheck` blocks in `docker-compose.yml` leveraging Python's built-in `urllib` to check container status without requiring external curl binaries.
- **Docker Hub Build & Publish Workflow**:
  - Added a GitHub Actions workflow `.github/workflows/docker-publish.yml` to build, tag (using SemVer and branch metadata), and publish the `jersobh/epochdb` image to Docker Hub automatically on tag and branch pushes.



---

## [0.2.0] - 2026-06-27

### Added
- **Multi-Role Server Roles**: Introduced `shard` (storage) and `coordinator` (routing gateway) modes configurable via `NODE_MODE` environment variable.
- **Consistent Hashing**: Added `ConsistentHashRing` utility for partitioning and distributing memory writes evenly across a variable list of storage shards.
- **Direct ID Prefix Routing**: Generated memory IDs are now prefixed with `shard{index}-`. Point lookups, updates, and deletes parse this prefix and route directly to the target shard without fanning out requests across the cluster.
- **Parallel Query Merging & Re-ranking**: The coordinator queries all shards in parallel and aggregates the results:
  - Vector searches are re-ranked by computed cosine similarity scores.
  - Entity graphs are merged and deduplicated.
  - Timelines are sorted chronologically.
  - Stats metrics are aggregated.
- **Extended REST APIs**: Added endpoints for `/get`, `/update`, `/delete`, `/get_timeline`, `/entity_graph`, and `/compact`.
- **Client SDK Extension**: Updated `client.py` with corresponding `get`, `update`, `delete`, `entity_graph`, `get_timeline`, and `compact` asynchronous methods.
- **Dockerization & Orchestration**:
  - Added `Dockerfile` using multi-stage builds and pre-installing `torch-cpu` to prevent download timeouts.
  - Created `docker-compose.yml` orchestrating a default local cluster of 3 shards and 1 coordinator gateway with persistent volumes.
- **Automated Integration Tests**: Added `test_cluster.py` to verify consistent hashing, direct routing, query merges, graph lookups, updates, and deletes.

### Fixed
- **PyTorch Image Build Failures**: Pre-installed `torch` CPU-only wheels inside the Docker container to bypass network timeouts and reduce container build overhead.
- **Typing Imports in Client**: Capitalized `Dict` and `List` type hint imports in `client.py` to preserve PEP-8 compliance and Python 3.12 compatibility.
