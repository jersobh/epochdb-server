# API Endpoint Reference

## GET `/healthz`
Liveness and readiness check.
- **Coordinator Mode**: Asserts all backend shards are online and responding.
- **Response**:
  ```json
  {
    "status": "healthy",
    "mode": "coordinator"
  }
  ```

---

## POST `/remember`
Ingests a new memory atom.
- **Payload**:
  ```json
  {
    "text": "Steve Jobs founded Apple.",
    "metadata": {
      "triples": [["Steve Jobs", "founded", "Apple"]]
    }
  }
  ```
- **Headers**: `X-API-Key` (Coordinator Mode) or `X-Internal-Token` (Shard Mode)
- **Response**:
  ```json
  {
    "status": "success",
    "id": "shard0-3a5a7bc..."
  }
  ```

---

## POST `/query`
Performs vector semantic search.
- **Payload**:
  ```json
  {
    "query": "Apple founders",
    "k": 3
  }
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "id": "shard0-3a5a7bc...",
        "text": "Steve Jobs founded Apple.",
        "metadata": {
          "triples": [["Steve Jobs", "founded", "Apple"]]
        },
        "score": 0.895
      }
    ]
  }
  ```

---

## POST `/delete`
Deletes a memory atom.
- **Payload**:
  ```json
  {
    "memory_id": "shard0-3a5a7bc...",
    "hard": true
  }
  ```

---

## GET `/stats`
Retrieves database diagnostics and resource metrics.
- **Response**:
  ```json
  {
    "mode": "coordinator",
    "memory_count": 12,
    "l1_size": 4,
    "system": {
      "cpu": 12.5,
      "ram": { "total": 16.0, "available": 12.4, "used": 3.6, "percent": 22.5 },
      "disk": { "total": 256.0, "available": 180.0, "used": 76.0, "percent": 29.6 }
    },
    "shards": {
      "http://shard0:8080": {
        "status": "healthy",
        "cpu": 8.4,
        "ram": { "percent": 15.2 },
        "disk": { "percent": 12.0 }
      }
    }
  }
  ```
