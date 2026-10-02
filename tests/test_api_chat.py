from fastapi.testclient import TestClient

from src.api import app


def test_chat_endpoint_uses_chatbot_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.api.answer_question",
        lambda q: {"question": q, "sql": "SELECT 1", "rows": [{"x": 1}], "error": None},
    )
    client = TestClient(app)
    resp = client.post("/chat", json={"question": "test?"})
    assert resp.status_code == 200
    assert resp.json()["rows"] == [{"x": 1}]


def test_chat_endpoint_requires_question() -> None:
    client = TestClient(app)
    resp = client.post("/chat", json={})
    assert resp.status_code == 400
