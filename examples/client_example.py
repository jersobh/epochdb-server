import os
import sys
import asyncio

from epochdb import AsyncRemoteEpochDB

async def main():
    db = AsyncRemoteEpochDB(host="127.0.0.1", port=8080, api_key="test-api-key-12345")
    
    try:
        # Store a memory asynchronously
        logger_meta = {"triples": [("Pollyanna", "married_to", "Jefferson")]}
        memory_id = await db.remember("Pollyanna is married to Jefferson.", metadata=logger_meta)
        print(f"Remember Response (ID): {memory_id}")
        
        # Get memory by ID
        mem = await db.get(memory_id)
        print(f"Get Memory: {mem}")
        if mem:
            print(f"  Text: {mem.text}")
            print(f"  Metadata: {mem.metadata}")
        
        # Query the remote database
        results = await db.query("Who is Pollyanna married to?", k=1)
        if results:
            print(f"Query Result: {results[0]}")
            print(f"  Text: {results[0].text}")
            print(f"  Score: {results[0].score}")
        
        # Get entity graph
        graph = await db.entity_graph("Pollyanna")
        print(f"Entity Graph: {graph}")
        
        # Update memory
        await db.update(memory_id, text="Pollyanna is happily married to Jefferson.")
        print(f"Update completed.")
        
        # Get timeline
        timeline = await db.get_timeline(entity_id="Pollyanna")
        print(f"Timeline: {timeline}")
        for item in timeline:
            print(f"  - {item.text} (created_at={item.created_at})")
        
        # Access database stats remotely
        stats = await db.stats()
        print(f"Stats: {stats}")

        
    finally:
        # Cleanly close the HTTP connection pool
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())
