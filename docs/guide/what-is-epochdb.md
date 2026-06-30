# What is EpochDB?

EpochDB is a high-concurrency sharded memory engine designed specifically for autonomous AI agents and workflows. It acts as an opinionated long-term memory layer that bridges semantic search capabilities with structured relational mappings.

## Core Concepts

EpochDB represents memory atoms with three main components:
1. **Payload (Text)**: The original factual statement or ingested document segment.
2. **Embedding**: A normalized vector representation (384-dimensional by default using the `all-MiniLM-L6-v2` model) to enable semantic vector queries.
3. **Triples**: Relational graph subject-predicate-object bindings (e.g., `["Albert Einstein", "born_in", "Germany"]`) to construct a distributed knowledge graph index.

---

## Memory Tiering

EpochDB uses a hybrid storage model to maximize speed and efficiency:

- **Hot Tier (L1 Cache)**: An in-memory cache implemented using Python dictionaries and SQLite memory logs. Writes are committed instantly to L1 and backed up asynchronously via a Write-Ahead Log (WAL).
- **Cold Tier (L2 Archive)**: Compacted, historical archives stored as compressed Parquet files on disk. The engine merges L1 memories into L2 during explicit compaction phases to reduce memory consumption while preserving search accuracy.
