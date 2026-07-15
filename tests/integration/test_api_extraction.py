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

FAKE_USER = {
    "sub": str(uuid.uuid4()),
    "app_metadata": {"tenant_id": str(uuid.uuid4()), "role": "analyst"},
    "email": "test@example.com",
}


@pytest.mark.asyncio
async def test_run_extraction_enqueues_job():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        from api.main import create_app
        from db.session import get_db
        from api.dependencies import get_queue

        app = create_app()

        mock_db = AsyncMock()
        mock_db.scalar = AsyncMock(return_value=None)
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        mock_queue = MagicMock()
        mock_queue.enqueue = MagicMock(return_value=None)

        async def override_get_db():
            yield mock_db

        def override_get_queue():
            return mock_queue

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_queue] = override_get_queue

        invoice_id = uuid.uuid4()
        job_id = uuid.uuid4()

        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            with patch("api.routers.extraction._get_invoice", return_value=MagicMock(id=invoice_id, status="ready", file_type="pdf")):
                with patch("api.routers.extraction._enqueue_extract", new=AsyncMock(return_value=job_id)):
                    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                        response = await client.post(
                            f"/api/v1/invoices/{invoice_id}/extract",
                            headers={"Authorization": "Bearer fake"},
                        )

    assert response.status_code == 200, response.text
    assert "job_id" in response.json()["data"]


@pytest.mark.asyncio
async def test_viewer_cannot_run_extraction():
    VIEWER_USER = {**FAKE_USER, "app_metadata": {**FAKE_USER["app_metadata"], "role": "viewer"}}
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        from api.main import create_app
        from db.session import get_db
        from api.dependencies import get_queue

        app = create_app()

        mock_db = AsyncMock()

        async def override_get_db():
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db

        with patch("api.dependencies.verify_supabase_jwt", return_value=VIEWER_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    f"/api/v1/invoices/{uuid.uuid4()}/extract",
                    headers={"Authorization": "Bearer fake"},
                )

    assert response.status_code == 403, response.text
