import os
import random
from locust import HttpUser, task, between

# -------------------------------------------------------------------------
# Dynamic Fact Generation Pool
# -------------------------------------------------------------------------
SUBJECTS = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", 
    "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Pete", "Quinn", 
    "Rose", "Sam", "Toby", "Alexander", "Sophia", "Zoe", "William", "Lucas"
]
PREDICATES = [
    "works_at", "lives_in", "loves", "hates", "owns", "created", "visited", 
    "studied", "teaches", "designed", "discovered", "wrote", "built", "repaired"
]
OBJECTS = [
    "Google", "London", "Paris", "Python", "Rust", "SQLite", "HNSW", "EpochDB", 
    "GitHub", "Docker", "Machine Learning", "Artificial Intelligence", "TypeScript",
    "Tokyo", "Berlin", "San Francisco", "React", "FastAPI", "Kubernetes", "Redis"
]

def generate_random_fact():
    sub = random.choice(SUBJECTS)
    pred = random.choice(PREDICATES)
    obj = random.choice(OBJECTS)
    text = f"{sub} {pred.replace('_', ' ')} {obj}."
    metadata = {"triples": [(sub, pred, obj)], "type": "synthetic_locust_test"}
    return text, metadata, sub, obj

class EpochDBUser(HttpUser):
    # Wait between 10ms and 100ms between tasks to simulate rapid back-to-back client requests
    wait_time = between(0.01, 0.1)
    
    # Class-level shared memory pool across user greenlets
    seeded_memories = []

    def on_start(self):
        # Resolve authentication configuration from environment
        self.api_key = os.getenv("API_KEY", "test-api-key-12345")
        self.internal_token = os.getenv("INTERNAL_AUTH_TOKEN", "test-internal-token-67890")
        self.shard_mode = os.getenv("SHARD_MODE", "false").lower() == "true"
        
        self.headers = {}
        if self.shard_mode:
            self.headers["X-Internal-Token"] = self.internal_token
        else:
            self.headers["X-API-Key"] = self.api_key

        # Seed initial memories to ensure read operations can be performed immediately
        for _ in range(5):
            self.seed_one_memory()

    def seed_one_memory(self):
        fact, meta, sub, obj = generate_random_fact()
        try:
            with self.client.post(
                "/remember", 
                json={"text": fact, "metadata": meta}, 
                headers=self.headers, 
                name="/remember (seed)", 
                catch_response=True
            ) as response:
                if response.status_code == 201:
                    new_id = response.json().get("id")
                    if new_id:
                        self.seeded_memories.append({"id": new_id, "subject": sub, "object": obj})
                        response.success()
                    else:
                        response.failure("Response missing memory ID")
                else:
                    response.failure(f"Failed to seed: HTTP {response.status_code}")
        except Exception as e:
            pass

    @task(35)
    def get_memory(self):
        if not self.seeded_memories:
            self.seed_one_memory()
            return
            
        mem = random.choice(self.seeded_memories)
        self.client.post(
            "/get", 
            json={"memory_id": mem["id"]}, 
            headers=self.headers, 
            name="/get"
        )

    @task(35)
    def query_memory(self):
        if not self.seeded_memories:
            self.seed_one_memory()
            return
            
        mem = random.choice(self.seeded_memories)
        query_str = f"Who studied {mem['object']}?"
        self.client.post(
            "/query", 
            json={"query": query_str, "k": 3}, 
            headers=self.headers, 
            name="/query"
        )

    @task(10)
    def remember_memory(self):
        fact, meta, sub, obj = generate_random_fact()
        with self.client.post(
            "/remember", 
            json={"text": fact, "metadata": meta}, 
            headers=self.headers, 
            name="/remember", 
            catch_response=True
        ) as response:
            if response.status_code == 201:
                new_id = response.json().get("id")
                if new_id:
                    self.seeded_memories.append({"id": new_id, "subject": sub, "object": obj})
                    response.success()
                else:
                    response.failure("Response missing memory ID")

    @task(10)
    def update_memory(self):
        if not self.seeded_memories:
            return
            
        mem = random.choice(self.seeded_memories)
        new_fact = f"{mem['subject']} updated study target."
        self.client.post(
            "/update", 
            json={"memory_id": mem["id"], "text": new_fact}, 
            headers=self.headers, 
            name="/update"
        )

    @task(2)
    def delete_memory(self):
        # Maintain a reasonable working set size
        if len(self.seeded_memories) <= 10:
            return
            
        mem = self.seeded_memories.pop(random.randint(0, len(self.seeded_memories) - 1))
        with self.client.post(
            "/delete", 
            json={"memory_id": mem["id"], "hard": False}, 
            headers=self.headers, 
            name="/delete", 
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
                # Restore
                self.seeded_memories.append(mem)

    @task(3)
    def entity_graph(self):
        if not self.seeded_memories:
            return
            
        mem = random.choice(self.seeded_memories)
        self.client.get(
            f"/entity_graph?entity_id={mem['subject']}&depth=1", 
            headers=self.headers, 
            name="/entity_graph"
        )

    @task(3)
    def get_timeline(self):
        if not self.seeded_memories:
            return
            
        mem = random.choice(self.seeded_memories)
        self.client.post(
            "/get_timeline", 
            json={"entity_id": mem["subject"]}, 
            headers=self.headers, 
            name="/get_timeline"
        )

    @task(2)
    def get_stats(self):
        self.client.get(
            "/stats", 
            headers=self.headers, 
            name="/stats"
        )
