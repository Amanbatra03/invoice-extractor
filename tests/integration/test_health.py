import os
import pytest
from unittest.mock import patch

# Patch env vars before any app imports
_TEST_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
    "REDIS_URL": "redis://localhost:6379",
    "SUPABASE_URL": "https://fake.supabase.co",
    "SUPABASE_ANON_KEY": "fake_anon",
    "SUPABASE_SERVICE_KEY": "fake_service",
    "SUPABASE_JWT_SECRET": "test_secret",
    "GOOGLE_API_KEY": "fake_google_key",
}


@pytest.mark.asyncio
async def test_health_returns_ok():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        from httpx import AsyncClient, ASGITransport
        from api.main import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_probe():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        from httpx import AsyncClient, ASGITransport
        from api.main import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["data"]["ready"] is True
