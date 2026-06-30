# Consistent Hashing & Sharding

To handle large-scale vector similarity searches and relationship graphs, EpochDB shards information across storage nodes.

## Consistent Hash Ring

The coordinator uses a Consistent Hashing Ring to distribute incoming memories based on their text payloads.
- **Node Replicas**: Each shard node has 100 virtual replica positions on the ring to guarantee uniform memory allocation and prevent database load imbalance.
- **Writing**: When the coordinator receives a write request on `/remember` without a predefined ID, it hashes the memory text, finds the closest active shard on the ring, prepends the shard index (e.g. `shard1-`), and forwards the write.

---

## Prefix-Based Query Routing

To maximize request speeds and avoid fanning out single-item queries to all nodes:
- **Direct Routing**: Queries like `/get`, `/update`, or `/delete` checking a prefixed ID (e.g. `shard0-5a3d7bc...`) are immediately routed directly to the exact target shard.
- **Broadcast Routing**: If an ID prefix is missing, the coordinator broadcasts the query to all shards in parallel and aggregates the responses.
