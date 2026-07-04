import os
import sys
import tempfile
import shutil
import asyncio
import pytest
import httpx
from fastapi.testclient import TestClient

# Setup environment variables BEFORE import
temp_dir = tempfile.mkdtemp(prefix="epochdb_sse_test_")
os.environ["NODE_MODE"] = "shard"
os.environ["INTERNAL_AUTH_TOKEN"] = "test-token-12345"
os.environ["STORAGE_DIR"] = temp_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.server import app, SSE_LISTENERS, broadcast_sse_event

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.mark.asyncio
async def test_sse_listeners_receive_broadcast():
    """
    Verifies that broadcast_sse_event successfully pushes event updates
    to all registered queues in the SSE_LISTENERS registry.
    """
    queue = asyncio.Queue()
    SSE_LISTENERS.append(queue)
    
    try:
        test_data = {"type": "write", "id": "test-atom-999"}
        await broadcast_sse_event("update", test_data)
        
        # Retrieve the event from our queue
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["event"] == "update"
        assert event["data"] == test_data
    finally:
        if queue in SSE_LISTENERS:
            SSE_LISTENERS.remove(queue)

@pytest.mark.asyncio
async def test_remember_endpoint_triggers_broadcast():
    """
    Verifies that calling /remember triggers a broadcast_sse_event call,
    putting a write notification into the active listener queue.
    """
    queue = asyncio.Queue()
    SSE_LISTENERS.append(queue)
    
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            remember_payload = {
                "text": "The boiling point of water is 100 degrees Celsius.",
                "metadata": {"source": "test_sse"}
            }
            headers = {"X-Internal-Token": "test-token-12345"}
            res = await client.post("/remember", json=remember_payload, headers=headers)
            assert res.status_code == 201
            
            # The write should have placed a notification in our queue
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert event["event"] == "update"
            assert event["data"]["type"] == "write"
            assert "id" in event["data"]
    finally:
        if queue in SSE_LISTENERS:
            SSE_LISTENERS.remove(queue)
