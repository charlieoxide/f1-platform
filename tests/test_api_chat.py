import pytest
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


import dataclasses  # noqa: E402

import src.api as api  # noqa: E402
from src.config import settings  # noqa: E402


@pytest.fixture
def fresh_limiter():
    api._REQUESTS.clear()
    yield
    api._REQUESTS.clear()


def _with_key(monkeypatch, key):
    monkeypatch.setattr(
        "src.api.settings", dataclasses.replace(settings, f1_api_key=key)
    )
    monkeypatch.setattr(
        "src.api.answer_question",
        lambda q: {"question": q, "sql": "SELECT 1", "rows": [], "error": None},
    )


def test_chat_is_rejected_without_or_with_the_wrong_key(
    monkeypatch, fresh_limiter
) -> None:
    _with_key(monkeypatch, "secret")
    client = TestClient(app)
    assert client.post("/chat", json={"question": "q"}).status_code == 401
    wrong = client.post("/chat", json={"question": "q"}, headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401


def test_chat_accepts_the_right_key_and_health_needs_none(
    monkeypatch, fresh_limiter
) -> None:
    _with_key(monkeypatch, "secret")
    client = TestClient(app)
    ok = client.post("/chat", json={"question": "q"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200


def test_rate_limit_blocks_the_request_after_the_limit(
    monkeypatch, fresh_limiter
) -> None:
    _with_key(monkeypatch, "secret")
    client = TestClient(app)
    headers = {"X-API-Key": "secret"}
    for _ in range(api._MAX_REQUESTS):
        assert (
            client.post("/chat", json={"question": "q"}, headers=headers).status_code
            == 200
        )
    assert (
        client.post("/chat", json={"question": "q"}, headers=headers).status_code == 429
    )
