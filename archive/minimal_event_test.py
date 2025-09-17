#!/usr/bin/env python3
"""
Minimal event producer test for OmniMemory RedPanda integration.
This bypasses complex model dependencies to test basic event production.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Try to import kafka client - install if needed
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    print("ERROR: kafka-python not installed. Run: pip install kafka-python")
    exit(1)


# Simple models for the test
class EventPayload(BaseModel):
    """Simple event payload for testing."""

    key: str
    data: str
    timestamp: str = None


class EventResponse(BaseModel):
    """Event production response."""

    success: bool
    event_id: str
    topic: str
    timestamp: str
    message: str


class MinimalEventProducer:
    """Minimal event producer for RedPanda testing."""

    def __init__(self, bootstrap_servers: str = "localhost:19092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Connect to RedPanda."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.bootstrap_servers],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda v: v.encode("utf-8") if v else None,
                retries=3,
                acks="all",  # Wait for all replicas
            )
            self.logger.info(f"Connected to RedPanda at {self.bootstrap_servers}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to RedPanda: {e}")
            return False

    def produce_event(
        self, topic: str, event_data: Dict[str, Any], key: str = None
    ) -> Dict[str, Any]:
        """Produce a single event to RedPanda."""
        if not self.producer:
            raise Exception("Producer not connected")

        # Create ONEX-compliant event structure
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        onex_event = {
            "event_id": event_id,
            "timestamp": timestamp,
            "source": "omnimemory-minimal-test",
            "event_type": "dev.omnibase.onex.cmd.omnimemory-store-memory.v1",
            "payload": event_data,
            "metadata": {"producer": "minimal_event_test", "version": "1.0.0"},
        }

        try:
            # Send event to RedPanda
            future = self.producer.send(
                topic=topic, value=onex_event, key=key or event_id
            )

            # Wait for confirmation
            record_metadata = future.get(timeout=10)

            result = {
                "success": True,
                "event_id": event_id,
                "topic": record_metadata.topic,
                "partition": record_metadata.partition,
                "offset": record_metadata.offset,
                "timestamp": timestamp,
            }

            self.logger.info(f"Event published successfully: {result}")
            return result

        except KafkaError as e:
            self.logger.error(f"Kafka error: {e}")
            raise Exception(f"Failed to publish event: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error publishing event: {e}")

    def close(self):
        """Close the producer connection."""
        if self.producer:
            self.producer.close()
            self.logger.info("Producer connection closed")


# FastAPI app for testing
app = FastAPI(
    title="OmniMemory Minimal Event Test",
    description="Minimal RedPanda event producer for testing",
    version="1.0.0",
)

# Global producer instance
event_producer = MinimalEventProducer()


@app.on_event("startup")
async def startup_event():
    """Connect to RedPanda on startup."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        success = event_producer.connect()
        if success:
            logger.info("✅ RedPanda connection established successfully")
        else:
            logger.warning("⚠️ Failed to connect to RedPanda - events will fail")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close producer on shutdown."""
    event_producer.close()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "OmniMemory Minimal Event Test API",
        "status": "running",
        "redpanda_connected": event_producer.producer is not None,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "redpanda_connection": "connected"
        if event_producer.producer
        else "disconnected",
        "bootstrap_servers": event_producer.bootstrap_servers,
    }


@app.post("/event/produce", response_model=EventResponse)
async def produce_test_event(payload: EventPayload):
    """Produce a test event to RedPanda."""
    try:
        # Prepare event data
        event_data = {
            "memory_key": payload.key,
            "memory_data": payload.data,
            "operation": "store",
            "timestamp": payload.timestamp or datetime.utcnow().isoformat(),
        }

        # Produce event
        result = event_producer.produce_event(
            topic="omnimemory-events", event_data=event_data, key=payload.key
        )

        return EventResponse(
            success=True,
            event_id=result["event_id"],
            topic=result["topic"],
            timestamp=result["timestamp"],
            message=f"Event published to partition {result['partition']} at offset {result['offset']}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to produce event: {str(e)}"
        )


@app.get("/event/test")
async def test_simple_event():
    """Produce a simple test event."""
    try:
        test_data = {
            "test_key": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "test_message": "Hello from OmniMemory minimal event test!",
            "test_number": 42,
        }

        result = event_producer.produce_event(
            topic="omnimemory-events", event_data=test_data, key="simple_test"
        )

        return {"message": "Test event produced successfully!", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test event failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "minimal_event_test:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info",
    )
