"""HTTP-level smoke tests that do not need a database."""

from fastapi.testclient import TestClient

from app.main import app
from app.services.database import database_service


def test_health_reports_database_state(monkeypatch):
    async def db_up() -> bool:
        return True

    async def db_down() -> bool:
        return False

    # Not used as a context manager on purpose: that would run the lifespan
    # hook, which pre-warms Postgres connections and blocks until they time out.
    client = TestClient(app)
    monkeypatch.setattr(database_service, "health_check", db_up)
    ok = client.get("/health")
    monkeypatch.setattr(database_service, "health_check", db_down)
    degraded = client.get("/health")

    assert ok.status_code == 200
    assert ok.json()["components"]["database"] == "healthy"
    assert degraded.status_code == 503
