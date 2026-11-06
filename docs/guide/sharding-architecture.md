# Consistent Hashing, Sharding & Replication

To balance memory indexing, CPU embedding pipelines, and fault tolerance, EpochDB supports both **horizontal sharding** and **high-availability replication sets**.

---

## 1. Consistent Hash Ring & Sharding

The coordinator uses a Consistent Hashing Ring to distribute incoming memories based on their text payloads.
- **Node Replicas**: Each shard node has 100 virtual replica positions on the ring to guarantee uniform memory allocation and prevent database load imbalance.
- **Writing**: When the coordinator receives a write request on `/remember` without a predefined ID, it hashes the memory text, finds the closest active shard on the ring, prepends the shard index (e.g. `shard1-`), and forwards the write.

---

## 2. Prefix-Based Query Routing

To maximize request speeds and avoid fanning out single-item queries to all nodes:
- **Direct Routing**: Queries like `/get`, `/update`, or `/delete` checking a prefixed ID (e.g. `shard0-5a3d7bc...`) are immediately routed directly to the exact target shard in $O(1)$ time.
- **Broadcast Routing**: If an ID prefix is missing, the coordinator broadcasts the query to all shards in parallel and aggregates the responses.

---

## 3. Sharding vs. Replication: High Availability

In a bare sharding topology (`shard0,shard1,shard2`), each shard owns a disjoint slice of the dataset. **If a shard node crashes or loses disk storage, the memories, procedural skills, and graph entities stored on that shard become unavailable**.

To guarantee **zero data loss and high availability (HA)**, EpochDB supports **replica sets**:

### Replication Syntax in `SHARD_NODES`

The coordinator parses `SHARD_NODES` using comma (`,`) for shard separation and plus (`+`) for replica clustering:

```bash
# 1. Pure Replicated Cluster (1 Shard, 3 Replicas — full dataset replicated across all 3 nodes):
SHARD_NODES="http://replica0:8080+http://replica1:8080+http://replica2:8080"

# 2. Bare Sharded Cluster (3 Shards, No Replication — each node holds 1/3 of data):
SHARD_NODES="http://shard0:8080,http://shard1:8080,http://shard2:8080"

# 3. Sharded + High Availability (2 Shards, each with 2 Replicas):
SHARD_NODES="http://s0a:8080+http://s0b:8080,http://s1a:8080+http://s1b:8080"
```

### Automatic Failover & Health Probes

- The coordinator continuously checks shard health in the background (`/healthz` and probe metrics).
- If a primary replica node fails or becomes overloaded, read requests are automatically dispatched to any surviving healthy replica in that group.
- Surving replicas report cumulative cluster telemetry and statistics even during single-node outages.

### Tunable Consistency Levels

Clients can specify replica write consistency per request via the `X-Consistency-Level` header:

```http
POST /remember HTTP/1.1
Host: localhost:8080
X-API-Key: test-api-key-12345
X-Consistency-Level: quorum
Content-Type: application/json

{
  "text": "Mission critical factual memory"
}
```

* **`one`** (default): Acknowledged as soon as any single replica in the group confirms the write.
* **`quorum`**: Acknowledged only when a majority ($\lfloor n/2 \rfloor + 1$) of replicas commit the write. Recommended for strong consistency without sacrificing availability.
* **`all`**: Strict consistency; write fails if even one replica in the group is unreachable.

---

## 4. Topology Decision Guide

| Requirement | Recommended Topology | Compose File |
| :--- | :--- | :--- |
| **Zero data loss on node failure, dataset < 500K memories** | Pure Replicas (Single Shard) | `docker-compose.replicas.yml` |
| **Max write throughput, horizontal RAM scaling** | Standard Sharded | `docker-compose.yml` |
| **Enterprise scale (millions of vectors) + High Availability** | Sharded + Replicated Sets | `docker-compose.sharded-ha.yml` |

