# Metrics & Health-Based Routing

EpochDB features intelligent, load-aware query and write routing based on real-time resource utilization.

## Resource Profilers

EpochDB storage shards collect performance statistics natively from the host Linux kernel without external python dependencies:
- **CPU Usage**: Extracted by computing delta ticks from `/proc/stat` over time.
- **RAM memory**: Extracted from `/proc/meminfo` metrics (`MemTotal` vs `MemAvailable`).
- **Storage**: Computed using native `os.statvfs` statistics of the storage mount volume.

---

## Coordinator Health Polling

The coordinator runs a non-blocking background loop polling shard health and resource usage every 5 seconds. Polled metrics are cached globally in `SHARD_METRICS_CACHE`.

---

## Load-Aware Routing Fallbacks

When writing memories via the consistent hash ring:
1. **Health Verification**: The coordinator checks if the primary hash-ring shard is online.
2. **Resource Load Evaluation**: If the shard's CPU, RAM, or storage utilization exceeds **90%**, or if it is offline:
   - The coordinator automatically redirects the write payload to the next healthy and underloaded shard in the cluster.
   - If all shards are under heavy load, it routes to any online shard, failing only if the entire cluster is offline.
