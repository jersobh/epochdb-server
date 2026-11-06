# Getting Started

Spin up your sharded memory database cluster in minutes.

## Prerequisites

- **Docker** and **Docker Compose**
- **Node.js** (optional, to run or build the documentation locally)

---

## Deploying the Cluster

EpochDB supports multiple cluster topologies depending on whether your priority is **zero data loss (high availability)**, **maximum horizontal scale (sharding)**, or both:

### Option 1: High-Availability Replicas (Recommended for Most Applications)

Every node stores 100% of memories, skills, and entity graphs. If any node goes down, zero data is lost:

```bash
docker compose -f docker-compose.replicas.yml up --build -d
```
- **Coordinator Gateway**: `http://localhost:8080`
- **Replicas 0, 1, 2**: `http://localhost:8081`, `http://localhost:8082`, `http://localhost:8083`

### Option 2: Standard Sharded Cluster (Max Horizontal Scale)

Data is split evenly across three independent shards using consistent hashing:

```bash
docker compose up --build -d
```
- **Coordinator Gateway**: `http://localhost:8080`
- **Shard 0, 1, 2**: `http://localhost:8081`, `http://localhost:8082`, `http://localhost:8083`

### Option 3: Sharded + High-Availability Replicas (Enterprise Scale & Redundancy)

Multi-shard architecture where each shard is protected by a replica pair:

```bash
docker compose -f docker-compose.sharded-ha.yml up --build -d
```
- **Coordinator Gateway**: `http://localhost:8080`
- **Shard 0 (Replicas A/B)**: `http://localhost:8081`, `http://localhost:8082`
- **Shard 1 (Replicas A/B)**: `http://localhost:8083`, `http://localhost:8084`

---

## Accessing the Dashboard

The built-in visualization panel is hosted directly on the coordinator gateway.

1. Open **EpochDB Studio** in your browser at `http://localhost:8080/visualize` (also available at `/`).
2. Authenticate using the default API key configured in `docker-compose.yml` (`test-api-key-12345`).
3. You will be greeted by the 3D Knowledge Graph Explorer dashboard.
