# Getting Started

Spin up your sharded memory database cluster in minutes.

## Prerequisites

- **Docker** and **Docker Compose**
- **Node.js** (optional, to run or build the documentation locally)

---

## Deploying the Cluster

EpochDB is fully configured to run inside a multi-container Docker cluster containing a coordinator gateway and three storage shards.

1. Locate the `docker-compose.yml` file in the root of the server repository.
2. Spin up the cluster using Docker Compose:
   ```bash
   docker-compose up --build
   ```
3. Once the database nodes finish warming up, the cluster endpoints will be active:
   - **Coordinator Gateway**: `http://localhost:8080` (External port)
   - **Shard 0**: `http://localhost:8081`
   - **Shard 1**: `http://localhost:8082`
   - **Shard 2**: `http://localhost:8083`

---

## Accessing the Dashboard

The built-in visualization panel is hosted directly on the coordinator gateway.

1. Open your web browser and navigate to `http://localhost:8080/visualize`.
2. Authenticate using the default API key configured in `docker-compose.yml` (`test-api-key-12345`).
3. You will be greeted by the 3D Knowledge Graph Explorer dashboard.
