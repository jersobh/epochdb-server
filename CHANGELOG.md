# Changelog

All notable changes to the EpochDB Distributed Server project will be documented in this file.

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
