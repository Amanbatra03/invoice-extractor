import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

_TEST_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
    "REDIS_URL": "redis://localhost:6379",
    "SUPABASE_URL": "https://fake.supabase.co",
    "SUPABASE_ANON_KEY": "fake_anon",
    "SUPABASE_SERVICE_KEY": "fake_service",
    "SUPABASE_JWT_SECRET": "test_secret",
    "GOOGLE_API_KEY": "fake_google_key",
}

ADMIN_USER = {
    "sub": str(uuid.uuid4()),
    "app_metadata": {"tenant_id": str(uuid.uuid4()), "role": "admin"},
    "email": "admin@example.com",
}

VIEWER_USER = {
    "sub": str(uuid.uuid4()),
    "app_metadata": {"tenant_id": str(uuid.uuid4()), "role": "viewer"},
    "email": "viewer@example.com",
}


def _make_mock_db():
    mock_db = AsyncMock()
    mock_db.scalar = AsyncMock(return_value=None)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    return mock_db


def _build_app(mock_db):
    from api.config import get_settings
    get_settings.cache_clear()
    from api.main import create_app
    from db.session import get_db

    app = create_app()

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest.mark.asyncio
async def test_unhandled_exception_returns_500_and_raises_alert():
    mock_db = _make_mock_db()
    with patch.dict(os.environ, _TEST_ENV):
        with patch("api.main.raise_alert", new=AsyncMock()) as mock_alert, \
             patch("api.main.get_session_factory") as mock_factory:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=AsyncMock())
            cm.__aexit__ = AsyncMock(return_value=False)
            mock_factory.return_value = MagicMock(return_value=cm)

            app = _build_app(mock_db)

            @app.get("/api/v1/_boom")
            async def _boom():
                raise RuntimeError("kaboom")

            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/v1/_boom")

    assert resp.status_code == 500
    body = resp.json()
    assert body["data"] is None
    assert body["error"] == "Internal server error"
    assert body["request_id"]  # request context middleware sets it
    mock_alert.assert_awaited_once()
    kwargs = mock_alert.await_args.kwargs
    assert kwargs["severity"] == "error"
    assert kwargs["source"] == "api"
    assert kwargs["event"] == "api.unhandled_exception"
    assert kwargs["detail"] == "kaboom"
    assert kwargs["context"]["path"] == "/api/v1/_boom"
    assert kwargs["context"]["method"] == "GET"


@pytest.mark.asyncio
async def test_exception_handler_survives_alert_failure():
    mock_db = _make_mock_db()
    with patch.dict(os.environ, _TEST_ENV):
        with patch("api.main.get_session_factory", side_effect=RuntimeError("db gone")):
            app = _build_app(mock_db)

            @app.get("/api/v1/_boom2")
            async def _boom2():
                raise RuntimeError("kaboom")

            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/v1/_boom2")

    assert resp.status_code == 500  # alert failure never breaks the 500 envelope
