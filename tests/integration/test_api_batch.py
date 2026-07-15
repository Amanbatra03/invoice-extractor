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

FAKE_TENANT_ID = str(uuid.uuid4())
FAKE_USER_ID = str(uuid.uuid4())

FAKE_USER = {
    "sub": FAKE_USER_ID,
    "app_metadata": {"tenant_id": FAKE_TENANT_ID, "role": "analyst"},
    "email": "test@example.com",
}


def _make_mock_db(job_id=None):
    mock_db = AsyncMock()
    job_id = job_id or uuid.uuid4()

    fake_job = MagicMock()
    fake_job.id = job_id

    mock_db.scalar = AsyncMock(return_value=None)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    # After refresh is called, simulate that job.id is set on the added object
    added_objects = []

    def capture_add(obj):
        added_objects.append(obj)
        if hasattr(obj, "type"):  # It's a Job
            obj.id = job_id

    mock_db.add = MagicMock(side_effect=capture_add)
    return mock_db


def _make_mock_queue():
    mock_q = MagicMock()
    mock_q.enqueue = MagicMock(return_value=None)
    return mock_q


def _build_app_with_overrides(mock_db, mock_queue):
    from api.config import get_settings
    get_settings.cache_clear()
    from api.main import create_app
    from db.session import get_db
    from api.dependencies import get_queue

    app = create_app()

    async def override_get_db():
        yield mock_db

    def override_get_queue():
        return mock_queue

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_queue] = override_get_queue
    return app


@pytest.mark.asyncio
async def test_batch_extract_enqueues_job():
    job_id = uuid.uuid4()
    mock_db = _make_mock_db(job_id)
    mock_queue = _make_mock_queue()
    ids = [str(uuid.uuid4()) for _ in range(3)]

    with patch.dict(os.environ, _TEST_ENV):
        app = _build_app_with_overrides(mock_db, mock_queue)

        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            with patch("api.routers.batch._all_invoices_exist", return_value=True):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/api/v1/batch/extract",
                        json={"invoice_ids": ids},
                        headers={"Authorization": "Bearer fake"},
                    )

    assert response.status_code == 200, response.text
    assert "batch_job_id" in response.json()["data"]


@pytest.mark.asyncio
async def test_batch_requires_at_least_one_invoice():
    mock_db = _make_mock_db()
    mock_queue = _make_mock_queue()

    with patch.dict(os.environ, _TEST_ENV):
        app = _build_app_with_overrides(mock_db, mock_queue)

        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/batch/extract",
                    json={"invoice_ids": []},
                    headers={"Authorization": "Bearer fake"},
                )

    assert response.status_code == 400, response.text
