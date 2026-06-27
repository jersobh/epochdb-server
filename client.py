import asyncio
import httpx
from typing import Optional, Dict, List, Any

class AsyncRemoteEpochDB:
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.base_url = f"http://{host}:{port}"
        # Use a persistent client session for connection pooling
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def remember(self, text: str, metadata: Dict = None) -> Dict:
        payload = {"text": text}
        if metadata:
            payload["metadata"] = metadata
            
        response = await self.client.post("/remember", json=payload)
        response.raise_for_status()
        return response.json()

    async def get(self, memory_id: str) -> Dict:
        payload = {"memory_id": memory_id}
        response = await self.client.post("/get", json=payload)
        response.raise_for_status()
        return response.json()

    async def query(self, query: str, k: int = 1, filters: Dict = None) -> List:
        payload = {"query": query, "k": k}
        if filters:
            payload["filters"] = filters
            
        response = await self.client.post("/query", json=payload)
        response.raise_for_status()
        return response.json()["results"]

    async def update(self, memory_id: str, text: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict:
        payload = {"memory_id": memory_id}
        if text is not None:
            payload["text"] = text
        if metadata is not None:
            payload["metadata"] = metadata
            
        response = await self.client.post("/update", json=payload)
        response.raise_for_status()
        return response.json()

    async def delete(self, memory_id: str, hard: bool = False) -> Dict:
        payload = {"memory_id": memory_id, "hard": hard}
        response = await self.client.post("/delete", json=payload)
        response.raise_for_status()
        return response.json()

    async def entity_graph(self, entity_id: str, depth: int = 2) -> Dict:
        response = await self.client.get("/entity_graph", params={"entity_id": entity_id, "depth": depth})
        response.raise_for_status()
        return response.json()

    async def get_timeline(self, entity_id: Optional[str] = None, start: Optional[float] = None, end: Optional[float] = None) -> List:
        payload = {}
        if entity_id is not None:
            payload["entity_id"] = entity_id
        if start is not None:
            payload["start"] = start
        if end is not None:
            payload["end"] = end
            
        response = await self.client.post("/get_timeline", json=payload)
        response.raise_for_status()
        return response.json()["memories"]

    async def stats(self) -> Dict:
        response = await self.client.get("/stats")
        response.raise_for_status()
        return response.json()

    async def compact(self) -> Dict:
        response = await self.client.post("/compact")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()


async def main():
    db = AsyncRemoteEpochDB(host="127.0.0.1", port=8080)
    
    try:
        # Store a memory asynchronously
        logger_meta = {"triples": [("Pollyanna", "married_to", "Jefferson")]}
        rem_res = await db.remember("Pollyanna is married to Jefferson.", metadata=logger_meta)
        print(f"Remember Response: {rem_res}")
        memory_id = rem_res["id"]
        
        # Get memory by ID
        mem = await db.get(memory_id)
        print(f"Get Memory: {mem}")
        
        # Query the remote database
        results = await db.query("Who is Pollyanna married to?", k=1)
        print(f"Query Result: {results[0]}")
        
        # Get entity graph
        graph = await db.entity_graph("Pollyanna")
        print(f"Entity Graph: {graph}")
        
        # Update memory
        up_res = await db.update(memory_id, text="Pollyanna is happily married to Jefferson.")
        print(f"Update Response: {up_res}")
        
        # Get timeline
        timeline = await db.get_timeline(entity_id="Pollyanna")
        print(f"Timeline: {timeline}")
        
        # Access database stats remotely
        stats = await db.stats()
        print(f"Stats: {stats}")
        
    finally:
        # Cleanly close the HTTP connection pool
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())
