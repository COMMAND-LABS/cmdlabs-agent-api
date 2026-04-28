"""Integration tests for API endpoints.

Uses FastAPI's TestClient with overridden dependencies — no real DB or LLM.
"""


def test_healthcheck(test_client):
    client, _ = test_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "OK!"}


def test_get_agent_not_found(test_client):
    client, mock_db = test_client
    mock_db.query.return_value.filter.return_value.first.return_value = None

    resp = client.get("/api/agents/999")
    assert resp.status_code == 404


def test_get_agent_found(test_client):
    from types import SimpleNamespace

    client, mock_db = test_client

    fake_account = SimpleNamespace(id=1, email="test@example.com")
    fake_agent = SimpleNamespace(id=42, name="TestBot", config={"data": {}}, account_id=1)

    call_count = [0]
    def query_side_effect(model):
        call_count[0] += 1
        result = type("MockQuery", (), {})()
        result.filter = lambda *a, **kw: result
        result.first = lambda: fake_account if call_count[0] == 1 else fake_agent
        result.join = lambda *a, **kw: result
        return result

    mock_db.query.side_effect = query_side_effect

    resp = client.get("/api/agents/42")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "TestBot"
    assert data["is_owner"] is True


def test_docs_endpoint(test_client):
    client, _ = test_client
    resp = client.get("/api/docs")
    assert resp.status_code == 200


def test_completion_requires_body(test_client):
    client, _ = test_client
    resp = client.post("/api/agents/1/completion")
    assert resp.status_code == 422


def test_stream_requires_body(test_client):
    client, _ = test_client
    resp = client.post("/api/agents/1/stream")
    assert resp.status_code == 422
