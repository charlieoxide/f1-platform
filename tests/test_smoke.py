import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api import app
from src.schemas import LapRecord


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_lap_record_validates_and_upcases_driver() -> None:
    rec = LapRecord(
        session_key="2024_Monaco_R",
        driver="ver",
        lap_number=1,
        lap_time_seconds=78.5,
    )
    assert rec.driver == "VER"


def test_lap_record_rejects_negative_lap_time() -> None:
    with pytest.raises(ValidationError):
        LapRecord(
            session_key="x",
            driver="VER",
            lap_number=1,
            lap_time_seconds=-1,
        )
