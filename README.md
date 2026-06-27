# EpochDB Distributed Server

[![Docker Pulls](https://img.shields.io/docker/pulls/jersobh/epochdb)](https://hub.docker.com/r/jersobh/epochdb)
[![Docker Image Version (latest by date)](https://img.shields.io/docker/v/jersobh/epochdb?sort=date)](https://hub.docker.com/r/jersobh/epochdb)
[![Publish](https://img.shields.io/github/actions/workflow/status/jersobh/epochdb-server/docker-publish.yml)](https://github.com/jersobh/epochdb-server/actions/workflows/docker-publish.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/jersobh/epochdb-server)](https://github.com/jersobh/epochdb-server/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EpochDB Distributed Server is an asynchronous, high-concurrency memory and vector database server designed for agentic workloads. It supports multi-node clustering with automatic horizontal sharding, consistent hashing write distribution, and direct prefix-based routing.


---

## Architectural Design

The cluster follows a MongoDB-style coordinator/gateway and storage shard architecture:

```
                  ┌──────────────────┐
                  │   HTTP Clients   │
                  └────────┬─────────┘
                           │ HTTP POST/GET
                           ▼
                  ┌──────────────────┐
                  │   Coordinator    │ (Gateway / Router Node)
                  └─┬──────┬──────┬──┘
                    │      │      │ HTTP (Routed / Broadcasted)
         ┌──────────┘      │      └──────────┐
         ▼                 ▼                 ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│    Shard 0     │ │    Shard 1     │ │    Shard 2     │ (Storage Nodes)
└────────────────┘ └────────────────┘ └────────────────┘
```

### 1. Node Roles (`NODE_MODE`)
A server instance runs in one of two modes depending on environment variables:
1. **Shard Mode (Storage Node)**:
   - Hosts a local `AsyncEpochDB` engine.
   - Manages physical SQLite files, WAL buffers, Parquet archives, and HNSW vector indexes locally.
   - Saves data under a configured `STORAGE_DIR`.
2. **Coordinator Mode (Router/Gateway Node)**:
   - Does not run a database locally.
   - Receives cluster configuration (URLs of the shards) via `SHARD_NODES`.
   - Distributes writes using consistent hashing and ID prefixing.
   - Parallelizes query searches and graph retrievals across shards, merging and ranking results.

### 2. Sharding & Routing
* **Consistent Hashing**: When writing memory through the coordinator, it hashes the memory `text` using MD5 consistent hashing to assign the write to a target shard node.
* **ID Prefixing**: Each generated ID is prefixed as `shard{index}-{uuid_hex}`. When fetching, updating, or deleting a memory, the coordinator parses the prefix to route the request **directly** to the target shard, avoiding expensive cluster-wide broadcasts. Broadcast is used as a fallback only if the prefix is missing or invalid.
* **Query Re-ranking**: For semantic queries, the coordinator queries all shards in parallel and aggregates, merges, and re-ranks the results by cosine similarity score before returning the top `k` candidates.
* **Graph Merging**: For `/entity_graph`, nodes and edges are retrieved from all shards and merged (set union) dynamically.

---

## Get Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (if running bare metal)

### Deploying the Cluster (Docker Compose)
A pre-configured 3-shard cluster and 1 coordinator gateway are defined in `docker-compose.yml`.

Run the following command to start the cluster:
```bash
docker compose up --build -d
```

This will run:
- `shard0` at `http://localhost:8081`
- `shard1` at `http://localhost:8082`
- `shard2` at `http://localhost:8083`
- `coordinator` gateway at `http://localhost:8080` (public interface)

### Running from Docker Hub Image

You can pull and run a node directly from the Docker Hub registry:

#### 1. Pull the Image
```bash
docker pull jersobh/epochdb:latest
```

#### 2. Run as a Shard Node (Storage Node)
Start a shard storage node, binding it to port `8080` on the host, and mounting a local directory to persist SQLite data, Parquet archives, and vector indexes:
```bash
docker run -d \
  --name epochdb-shard \
  -p 8080:8080 \
  -e NODE_MODE=shard \
  -e STORAGE_DIR=/data \
  -e INTERNAL_AUTH_TOKEN=your-secure-internal-token \
  -v /absolute/path/to/local/data:/data \
  jersobh/epochdb:latest
```

#### 3. Run as a Coordinator Node (Gateway Router)
Start a gateway router node that distributes queries across backend shards:
```bash
docker run -d \
  --name epochdb-coordinator \
  -p 8080:8080 \
  -e NODE_MODE=coordinator \
  -e SHARD_NODES=http://shard0-ip:8080,http://shard1-ip:8080 \
  -e API_KEY=your-client-api-key \
  -e INTERNAL_AUTH_TOKEN=your-secure-internal-token \
  jersobh/epochdb:latest
```

### Local Configuration

Environment variables used by the server:
- `NODE_MODE`: `"shard"` (default) or `"coordinator"`.
- `SHARD_NODES`: Comma-separated list of backend shard URLs (e.g. `http://shard0:8080,http://shard1:8080`). Required in coordinator mode.
- `STORAGE_DIR`: Local data storage directory (default: `./shared_memory`).

---

## API Reference

All requests and responses use JSON.

### 1. `POST /remember`
Ingests a new memory.
* **Payload**:
  ```json
  {
    "text": "Factual text to store.",
    "metadata": { "optional": "metadata", "triples": [["subject", "predicate", "object"]] },
    "id": "optional-predefined-id"
  }
  ```
* **Response**:
  ```json
  { "status": "success", "id": "shard1-46bb87d8..." }
  ```

### 2. `POST /get`
Retrieves a memory by ID.
* **Payload**: `{ "memory_id": "shard1-46bb87d8..." }`
* **Response**:
  ```json
  {
    "id": "shard1-46bb87d8...",
    "payload": "Factual text to store.",
    "payload_type": "text",
    "embedding": [0.012, -0.054, ...],
    "triples": [["subject", "predicate", "object"]],
    "created_at": 1719542000.123,
    "access_count": 1,
    "epoch_id": "epoch_active",
    "metadata": { "optional": "metadata" }
  }
  ```

### 3. `POST /query`
Performs semantic search.
* **Payload**:
  ```json
  {
    "query": "search query string",
    "k": 5,
    "filters": { "optional": "metadata-filters" }
  }
  ```
* **Response**:
  ```json
  {
    "results": [
      {
        "id": "shard1-46bb87d8...",
        "text": "Factual text to store.",
        "metadata": { "optional": "metadata" },
        "created_at": 1719542000.123,
        "score": 0.892
      }
    ]
  }
  ```

### 4. `POST /update`
Updates an existing memory's text or metadata.
* **Payload**:
  ```json
  {
    "memory_id": "shard1-46bb87d8...",
    "text": "new text payload",
    "metadata": { "updated": "metadata" }
  }
  ```
* **Response**: `{ "status": "success" }`

### 5. `POST /delete`
Deletes a memory.
* **Payload**:
  ```json
  {
    "memory_id": "shard1-46bb87d8...",
    "hard": false
  }
  ```
* **Response**: `{ "status": "success" }`

### 6. `GET /entity_graph`
Retrieves the combined entity graph.
* **Parameters**: `entity_id` (string, required), `depth` (integer, default: 2)
* **Response**:
  ```json
  {
    "nodes": ["Marie Curie", "radium"],
    "edges": [
      {
        "source": "Marie Curie",
        "target": "radium",
        "predicate": "discovered",
        "memory_id": "shard0-abc..."
      }
    ]
  }
  ```

### 7. `POST /get_timeline`
Gets chronological memories across all nodes.
* **Payload**: `{ "entity_id": "optional", "start": 0.0, "end": 1719542000.0 }`
* **Response**: `{ "memories": [...] }`

### 8. `GET /stats`
Retrieves aggregated metrics.
* **Response**:
  ```json
  {
    "memory_count": 12,
    "l1_size": 2,
    "l2_size": 10,
    "entity_count": 5
  }
  ```

### 9. `POST /compact`
Compact all cluster nodes.
* **Response**: `{ "status": "compaction completed" }`

---

## Client SDK Usage

Use the provided `client.py` wrapper to interact with the database:

```python
import asyncio
from epochdb import AsyncRemoteEpochDB

async def main():
    # Connect to the coordinator gateway
    db = AsyncRemoteEpochDB(host="127.0.0.1", port=8080, api_key="test-api-key-12345")
    
    # Store a memory
    memory_id = await db.remember(
        "Marie Curie discovered radium.", 
        metadata={"triples": [("Marie Curie", "discovered", "radium")]}
    )
    print(f"Stored: {memory_id}")
    
    # Semantic Query
    res = await db.query("Marie Curie discoveries", k=1)
    if res:
        print(f"Query match: {res[0].text} (Score: {res[0].score})")
    
    # Entity Graph
    graph = await db.entity_graph("Marie Curie")
    print(f"Graph: {graph}")
    
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](file:///home/jeff/Projects/epochdb-server/LICENSE) file for details.

