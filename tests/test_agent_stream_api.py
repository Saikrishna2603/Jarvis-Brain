from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from jarvis_brain.routes.agent_stream import get_agent_stream_service
from jarvis_brain.app import app
from jarvis_platform.schemas.agent_stream import AgentStreamEnvelope, AgentStreamEventType
from jarvis_brain.agents.agent_stream_service import AgentStreamService


class FiniteStreamService:
    async def stream_events(
        self, include_initial_snapshot: bool = True
    ) -> AsyncIterator[str]:
        connected = AgentStreamEnvelope(
            stream_event_id="stream-connected",
            event_type=AgentStreamEventType.CONNECTED,
        )
        yield AgentStreamService.format_sse(connected)
        if include_initial_snapshot:
            snapshot = AgentStreamEnvelope(
                stream_event_id="stream-snapshot",
                event_type=AgentStreamEventType.SNAPSHOT,
                data={
                    "active_agents": [],
                    "completed_agents": [],
                    "failed_agents": [],
                    "archived_agents": [],
                    "recent_events": [],
                    "graph_nodes": [],
                    "graph_edges": [],
                    "metadata": {},
                },
            )
            yield AgentStreamService.format_sse(snapshot)


client = TestClient(app)


def test_agent_stream_status() -> None:
    response = client.get("/agents/stream/status")

    assert response.status_code == 200
    assert response.json()["transport"] == "sse"
    assert response.json()["storage"] == "in_memory"
    assert "connected_clients" in response.json()


def test_agent_stream_endpoint_has_sse_headers_and_initial_events() -> None:
    app.dependency_overrides[get_agent_stream_service] = FiniteStreamService
    try:
        response = client.get("/agents/stream/events?include_snapshot=true")
    finally:
        app.dependency_overrides.pop(get_agent_stream_service, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: connected" in response.text
    assert "event: snapshot" in response.text


def test_agent_stream_without_snapshot_is_finite_and_safe() -> None:
    app.dependency_overrides[get_agent_stream_service] = FiniteStreamService
    try:
        response = client.get("/agents/stream/events?include_snapshot=false")
    finally:
        app.dependency_overrides.pop(get_agent_stream_service, None)

    assert response.status_code == 200
    assert "event: connected" in response.text
    assert "event: snapshot" not in response.text
    assert "sk-test" not in response.text


def test_existing_lifecycle_rest_api_still_works() -> None:
    assert client.get("/agents/lifecycle/status").status_code == 200
    assert client.get("/agents/lifecycle/snapshot").status_code == 200
