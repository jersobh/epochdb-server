# Changelog

All notable changes to the EpochDB Distributed Server project will be documented in this file.

## [0.14.0] - 2026-11-07
### Added
- **Unified "All Namespaces" (`*` / `all`) Scope Aggregation**:
  - Added native wildcard namespace support (`*` and `all`) to `/entity_graph`, `/stats`, `/export`, `/skills`, and `/entity_types`.
  - Aggregates disparate partitions within a tenant into a unified cross-partition 2D/3D knowledge graph while strictly preserving tenant isolation boundaries.
  - Dynamically merges and deduplicates relational triples, typed entities, procedural skills, and user profile memories across active and cold database partitions.
  - Automatic tenant-wide cache invalidation: write mutations to any single namespace automatically purge the tenant's wildcard cache.
- **Tenant and Namespace Discovery & Autocomplete in Studio UI**:
  - Connected Studio Scope modal (`AuthModal.tsx`) to `/databases` discovery endpoint.
  - Dynamically autocompletes available tenants and namespaces with input datalists and quick select options.
  - Supports selecting `*` (all namespaces) directly from scope settings.

## [0.13.2] - 2026-11-06
### Added
- **Embedding & Extraction Model Support**:
  - Integrated support and benchmarks for `jersobh/epochdb-embed-0.6b` (1024-dimension, ~600M parameters, Matryoshka and MultipleNegativesRankingLoss trained) and `jersobh/epochdb-extractor-0.8b` (fine-tuned Qwen3.5-0.8B for canonical relational triple extraction).
  - Added benchmarking script `benchmark_models.py` evaluating model load time, single-sentence and batch encoding latency, retrieval accuracy, distinction margins, and triple extraction quality.
  - Added runtime compatibility alias for `sentence_transformers.base.modules.normalize` allowing sentence-transformers 6.x-trained models to run on 5.x environments without import failures.
  - Extended HuggingFace fact extraction to support decoder-only CausalLM models (`AutoModelForCausalLM` fallback) and resilient Python dict literal parsing (`ast.literal_eval`).
- **High-Availability & Replication Topologies**:
  - Added `docker-compose.replicas.yml` for active-active multi-replica cluster setups and `docker-compose.sharded-ha.yml` for sharded clusters with per-shard replica sets.
  - Added detailed architectural documentation on sharding vs replica tradeoffs and fault-tolerant deployment modes in `docs/guide/sharding-architecture.md` and `docs/guide/getting-started.md`.

### Fixed
- **Entity Graph Traversal Depth & Co-Occurrence Leakage**:
  - Fixed an issue where querying `/entity_graph` with an `entity_id` (e.g. `Alan Turing`) at `depth=2` leaked distant, unrelated skills (such as `Linux Kernel` and `Git Version Control`) into the subgraph.
  - Resolved root cause where raw document-level `co_occurs_with` edges bridged memory atoms, and stripped bridge edges left disconnected island subgraphs in `all_edges` whose endpoints were inadvertently added to `final_nodes`.
  - Implemented strict BFS distance pruning (`_prune_subgraph_by_depth`) on both coordinator and shard endpoints to discard any nodes and edges beyond the requested traversal radius.
  - Removed artificial depth clamping on shards (`min(depth, 2)`) and coordinator (`min(depth, 4)`), allowing the UI traversal depth slider to function across its full 1–10 hop range.
  - Added automated test suite coverage (`test_entity_graph_traversal_depth` in `tests/test_server_api.py`).
- **UI Graph Explorer Filtering**:
  - Fixed race condition in `ExplorerDock.tsx` and `App.tsx` where picking an entity autocomplete suggestion dropped the selected filter value.
  - Fixed procedural skill node demotion to generic `entity` in `ui/src/lib/graph.ts` and enabled skill memory type recognition in `src/type_discovery.py`.

## [0.13.1] - 2026-11-04
### Fixed
- **Coordinator & Shard Stats Memory Count Reporting**: Fixed an issue where coordinator `GET /stats` returned `memory_count == 0` due to `_LAST_SHARD_DB_STATS` remaining 0 on shards and coordinator serving stale polling cache on live requests.
  - Implemented `get_shard_db_stats()` to dynamically snapshot live atom counts directly from in-memory tier metadata (`hot_tier.atoms` and `cold_tier.get_total_atoms()`) with zero model latency or lock contention.
  - Ensured coordinator live `/stats` requests (when `probe=False`) query shards directly and dynamically update `SHARD_METRICS_CACHE` with fresh metrics while preserving cached responses for health probes (`probe=True`).
  - Improved coordinator replica group aggregation with an `accounted_groups` accumulator so surviving replicas properly report cluster statistics even during single-shard outages.

## [0.13.0] - 2026-10-22
### Added
- **`GET /stats?probe=1`**: Lightweight shard metrics (CPU/RAM/disk + last-known counts) that never open the DB or run embeddings. Coordinator `poll_shards_loop` uses this so health polls stay green under MiniLM load.
- **`DEFAULT_TENANT` / `DEFAULT_NAMESPACE`**: Optional shard boot pre-warm of an app partition (e.g. Percepta `global`/`default`) so the first request does not cold-load a second MiniLM instance.
- **`COORDINATOR_GRAPH_TIMEOUT`** (default `25s`): Bound entity-graph fan-out separately from the 120s write path; partial graphs report `partial` + `failed_shards`.
- **`GRAPH_HUB_CONCURRENCY`**: Bounded parallel hub BFS for cluster-wide graphs (default hub limit lowered to 12).
- **Entity type registry** (`src/type_registry.py`): Apps can predefine or create types via `GET/POST /entity_types`, `PUT /entity_types/seed`, `DELETE /entity_types/{id}` (per tenant/namespace). Persisted under `TYPE_REGISTRY_DIR` / `STORAGE_DIR`.
- **Type auto-discovery** (`src/type_discovery.py`): Open-vocab types from stored `is_a` / type triples (engine API or hot-tier scan). Merged with registered types in `/discovered_types` and `/entity_types`.
- **Suggested seed catalog** (`src/ontology.py`): Optional ~22-type seed (`person`, `organization`, `product`, …) via `use_default_seed` or app-supplied lists — not a hard clamp on discovery.
### Changed
- **Coordinator `GET /stats`**: Serves from `SHARD_METRICS_CACHE` when populated (same philosophy as `/healthz`); live fan-out uses probe mode with a short timeout.
- **Gunicorn timeout** default raised to `180s` via `GUNICORN_TIMEOUT` (workers remain `1` — EpochDB `FileLock` is exclusive per `/data`).
- **Studio type legend**: Uses registered ∪ discovered types; `holds_role` no longer invents types from job titles; sentence/colon junk → `entity` (hidden by default).
- **Overview `/entity_graph`**: Cheap KG neighbor walk (no hub×BFS) plus `typed_nodes` from id heuristics (`person_*` → person).
### Fixed
- **Coordinator `/import` false success on replica failure (#41)**: Inspect replica group responses during coordinator `/import` instead of discarding gather results and unconditionally reporting success. Items are counted only when replica writes succeed according to consistency requirements (at least one or quorum); failed items are recorded under `failed` and `errors`, returning accurate imported counts and error details to eliminate silent data loss.
- **Empty entity-graph error logs**: Failures now use `format_shard_error()` so `ReadTimeout('')` is visible.
- **Worker SIGABRT under load**: Background stats probes no longer compete with embeds for the single worker’s DB/event loop, reducing gunicorn `WORKER TIMEOUT` kills and WAL replay storms.
- **Studio empty graph under MiniLM load**: Hub BFS overview wedged `workers=1` shards; replaced with index walk.

## [0.12.0] - 2026-10-21
### Added
- **Core Engine Upgrade**: Requires `epochdb >= 1.12.0` for open-vocab type discovery and typed entity graphs.
- **Discovered Types API (`GET /discovered_types`)**: Returns the union of type objects and predicates from stored triples (shard-local or coordinator-merged). No fixed ontology / enum registry.
- **Typed entity graph payloads**: `/entity_graph` may include `node_types` and `typed_nodes` when the engine discovers type-defining triples.
- **`EPOCHDB_EXTRACTION_MODEL`**: Explicit extraction model env (default `local`) so host `GEMINI_API_KEY` / cloud extractors are not auto-selected on shards.
### Changed
- **Studio open-vocab types**: Legend and filters derive from discovered graph types; dynamic styles for unknown types; default node type is `entity`.
- **Studio tenant headers**: Client sends `X-Tenant` (and legacy `X-Tenant-ID`) to match server auth.
- **Compose / `.env.example`**: Document `EPOCHDB_EXTRACTION_MODEL=local`; remove `GEMINI_API_KEY` from shard compose env.
### Fixed
- **Coordinator `/entity_graph?entity_id=` latency**: Replaced hop×frontier×shard depth-1 BFS with a single fan-out per shard at full depth (shards BFS locally). Cuts person-graph load from tens of seconds to one round-trip under embed load.
- **Coordinator `/healthz` under embed load**: Uses the background shard metrics cache instead of a live 2s fan-out probe (configurable `COORDINATOR_HEALTHZ_TIMEOUT` only for uncached shards). Prevents Docker/compose health flaps when single-worker shards are CPU-bound on MiniLM.
- **Compose networking**: Explicit shared network + `depends_on: service_healthy` so the coordinator only starts after shards are reachable as `shard0`/`shard1`/`shard2` (avoids `No address associated with hostname` and cold-start probe timeouts while the embedding model loads). Healthcheck `start_period` raised to 90s for first-boot model download.
- **Seed Sample disabled** in Explorer dock so Percepta / Studio does not auto-load demo graphs.

## [0.11.0] - 2026-10-13
### Added
- **Core Engine Upgrade**: Upgraded to `epochdb >= 1.11.2`, unlocking native batch ingestion, high-speed procedural skill index, user profile memory types, hot summary snapshots, and neuro-symbolic propose/LCAG validation.
- **Batch Memory Ingestion (`POST /remember_batch`)**: Ingest multiple memories in a single atomic network roundtrip. In coordinator mode, batch items are grouped by consistent-hash target shard and distributed in parallel.
- **In-Place Atomic Replacement (`POST /replace`)**: Atomically replace an existing memory's text, triples, and metadata with durable write-ahead log updates and cache invalidation.
- **Neuro-Symbolic LCAG Validation (`POST /propose`)**: Two-phase state commit endpoint that verifies proposed memory transitions against linear-causal DAG constraints before committing.
- **Relational Multi-Hop Retrieval (`POST /multi_hop`)**: Relational search traversing the knowledge graph over multiple expansion hops (`hops` parameter) with candidate ranking.
- **Procedural Skills Index & Endpoints**:
  - `POST /skills/remember`: Register agent procedural skills and operational recipes.
  - `GET /skills/{skill_name}` & `POST /skills/get`: Retrieve procedural skills by name via fast `SkillIndex`.
  - `GET /skills`: List all procedural skills registered across the cluster.
- **User Profile Dialectic Facts & Snapshots**:
  - `POST /profiles/remember`: Deterministic routing and storage of user identity preferences under `MemoryType.PROFILE`.
  - `GET /profiles/{user_id}` & `POST /profiles/get`: Retrieve profile facts with fallback to zero-threshold semantic filtering.
  - `GET /summary/snapshot` & `POST /summary/snapshot`: Aggregated compact textual summaries of active hot memories and profile facts.
- **Extended Query Options**: `POST /query` now supports `cold_search_mode`, `min_score` thresholding, and hybrid RRF text queries.
- **Pending Extractions Metric**: `/stats` now reports `pending_extractions` in shard mode.
### Changed
- **EpochDB Studio (React)**: Replaced the monolithic `visualize.html` template with a Vite + React app under `ui/`, preserving look, feel, and feature parity (Explorer dock, Memory Inspector, auth/theme, SSE, seed, 2D/3D graphs, import/export). FastAPI serves the built SPA from `src/static/ui/` at `/` and `/visualize` (with injected `__EPOCHDB_SERVER_CONFIG__` and `/assets` static mount); Docker builds the UI in a multi-stage image.
### Fixed
- **Studio graph remount crash**: Toggling entity-type filters (all off → one on) no longer throws `removeChild` `NotFoundError`; 2D/3D canvases stay mounted and update via stable hosts / vis `DataSet`.
- **Studio 3D navigation**: Orbit/zoom controls work after switching from 2D — the WebGL canvas is no longer initialized under `display: none`.

## [0.10.7] - 2026-08-03
### Added
- **Per-Shard Exponential Backoff**: Introduced adaptive exponential backoff in `poll_shards_loop` for unreachable or timing-out shards to eliminate phantom CPU overhead from repeated TCP connection handshakes and teardowns.
- **Configurable Coordinator Polling Parameters**: Added `COORDINATOR_POLL_INTERVAL` (default `5s`) and `COORDINATOR_MAX_BACKOFF` (default `60s`) environment variables for coordinator health probe tuning.

## [0.10.6] - 2026-07-26
### Changed
- **Dependency**: Requires `epochdb>=1.8.4`, including robust HNSW queries after hard deletes and improved retrieval of three-character conversational entities.

## [0.10.5] - 2026-07-26
### Fixed
- **Cross-shard write routing**: Coordinator writes now stay in their consistent-hash owner group. Saturated or unavailable owners return `503` with `Retry-After` instead of stranding records on a fallback shard.
### Added
- **Embedding backpressure**: Per-shard bounded embedding capacity (`SHARD_EMBED_CONCURRENCY`, default `1`) and queue limit (`SHARD_EMBED_QUEUE_LIMIT`, default `8`). Saturated shards return a retryable `503`.
### Changed
- **Dependency**: Requires `epochdb>=1.8.3` for checkpoint-safe WAL markers, durable hard deletes, and vector-index update maintenance.

## [0.10.4] - 2026-07-26
### Fixed
- **Coordinator write/query timeouts under load**: Default coordinator→shard HTTP timeout raised from 30s to 120s (`COORDINATOR_HTTP_TIMEOUT`), matching gunicorn worker timeout. Concurrent MiniLM embeds on small VPS CPUs regularly exceeded 30s, causing empty `Last error:` / `Error querying shard:` logs and 500s on `/remember`.
- **False offline marking**: Timeouts no longer mark shards `offline` (only hard `ConnectError`), avoiding cascading routing failures while shards are merely busy.
- **Stats poll flapping**: `/stats` probe timeout raised (`COORDINATOR_STATS_TIMEOUT`, default 15s). Transient stats failures keep the prior health status instead of flipping every shard to `unhealthy` (which triggered `All nodes unhealthy/offline` despite `healthz` 200).
- **Opaque exception logs**: Shard fan-out errors now use `repr()` / response body when `str(exc)` is empty (e.g. bare `TimeoutException`).
- **Partial query results were cached**: `/query` responses missing shard answers are no longer stored in the coordinator read cache, so results recover as soon as shards do.
### Added
- **`partial` / `failed_shards` fields on `/query`**: Coordinator responses now say when one or more shards failed to answer, so clients can distinguish "no matches" from degraded fan-out.
- **`LOG_LEVEL` env var** for the server root logger.
### Changed
- **Much quieter logs**: httpx/httpcore request lines silenced (WARNING+), per-5s poll and SSE broadcast logs demoted to DEBUG, per-write cache-invalidation logs demoted to DEBUG, and tqdm/HF progress bars disabled in the Docker image (`TQDM_DISABLE=1`, `HF_HUB_DISABLE_PROGRESS_BARS=1`).

## [0.10.3] - 2026-07-26
### Changed
- **Dependency**: Requires `epochdb>=1.8.2` (LocalFactExtractor no longer promotes predicates into co-occurrence graph nodes).

## [0.10.2] - 2026-07-26
### Fixed
- **Portable hnswlib boot**: Dockerfile force-rebuilds `hnswlib` from source with `HNSWLIB_NO_NATIVE=1` and `--no-binary=hnswlib`, preventing Illegal Instruction crashes on CPUs without AVX-512.
- **Stale lock cleanup**: Startup recursively clears `*.lock` under `/data` before gunicorn boots.
### Changed
- **Merge AUTO_EXTRACT with provided triples**: `/remember` (fixed-id replace path) discovers entity co-occurrence links when `AUTO_EXTRACT=true` and merges them with caller-supplied structural triples.
- **Dependency**: Requires `epochdb>=1.8.1` from PyPI (FileLock re-entrancy + triple merge).

## [0.10.0] - 2026-07-25
### Added
- **Scope Autocomplete API**: Added `GET /databases` endpoint to list available tenant and namespace partitions for UI autocompletion.
- **Visualization Version & Scope UI**:
  - Integrated datalist autocompletion for Tenants and Namespaces in `visualize.html`.
  - Exposed server version in `/healthz` response and added server version badge in `visualize.html` header bar.

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
