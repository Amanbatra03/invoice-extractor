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


def _make_mock_db():
    mock_db = AsyncMock()
    mock_db.scalar = AsyncMock(return_value=None)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.delete = AsyncMock()
    return mock_db


def _build_app_with_overrides(mock_db):
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
async def test_create_webhook():
    mock_db = _make_mock_db()

    webhook_id = uuid.uuid4()
    tenant_id = uuid.UUID(ADMIN_USER["app_metadata"]["tenant_id"])
    created_at_val = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)

    def capture_add(obj):
        obj.id = webhook_id
        obj.tenant_id = tenant_id
        obj.url = "https://example.com/hook"
        obj.events = ["extraction.completed"]
        obj.secret = "s3cr3t"
        obj.active = True
        obj.created_at = created_at_val

    mock_db.add = MagicMock(side_effect=capture_add)

    async def fake_refresh(obj):
        pass

    mock_db.refresh = AsyncMock(side_effect=fake_refresh)

    with patch.dict(os.environ, _TEST_ENV):
        app = _build_app_with_overrides(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=ADMIN_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/webhooks",
                    json={"url": "https://example.com/hook", "events": ["extraction.completed"], "secret": "s3cr3t"},
                    headers={"Authorization": "Bearer fake"},
                )

    assert response.status_code == 200, response.text
    assert response.json()["data"]["url"] == "https://example.com/hook"


@pytest.mark.asyncio
async def test_analyst_cannot_create_webhook():
    ANALYST = {**ADMIN_USER, "app_metadata": {**ADMIN_USER["app_metadata"], "role": "analyst"}}
    mock_db = _make_mock_db()

    with patch.dict(os.environ, _TEST_ENV):
        app = _build_app_with_overrides(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=ANALYST):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/webhooks",
                    json={"url": "https://example.com/hook", "events": ["extraction.completed"], "secret": "s"},
                    headers={"Authorization": "Bearer fake"},
                )

    assert response.status_code == 403
