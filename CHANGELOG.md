# Changelog

All notable changes to the EpochDB Distributed Server project will be documented in this file.

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
