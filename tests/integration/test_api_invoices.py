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


def _make_mock_db():
    """Return a mock AsyncSession that returns None for scalar() and empty for execute()."""
    mock_db = AsyncMock()
    mock_db.scalar = AsyncMock(return_value=None)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()
    mock_db.delete = AsyncMock()
    return mock_db


def _make_mock_queue():
    mock_q = MagicMock()
    mock_q.enqueue = MagicMock(return_value=None)
    return mock_q


def _build_app_with_overrides(mock_db, mock_queue):
    with patch.dict(os.environ, _TEST_ENV):
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
async def test_upload_invoice_returns_job_id():
    mock_db = _make_mock_db()
    mock_queue = _make_mock_queue()

    # After flush/refresh, invoice and job need realistic UUIDs
    invoice_id = uuid.uuid4()
    job_id = uuid.uuid4()

    fake_invoice = MagicMock()
    fake_invoice.id = invoice_id

    fake_job = MagicMock()
    fake_job.id = job_id

    # scalar returns None (no duplicate), then after add/flush/refresh we set invoice.id
    mock_db.scalar = AsyncMock(return_value=None)

    # Capture what gets added so we can set ids after add()
    added_objects = []

    def capture_add(obj):
        added_objects.append(obj)
        # Assign ids when objects are added
        if hasattr(obj, 'sha256'):  # It's an Invoice
            obj.id = invoice_id
        elif hasattr(obj, 'type'):  # It's a Job
            obj.id = job_id

    mock_db.add = MagicMock(side_effect=capture_add)

    app = _build_app_with_overrides(mock_db, mock_queue)

    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        with patch("api.routers.invoices.upload_file", return_value="path/to/file.pdf"):
            with patch("api.routers.invoices._enqueue_ingest", new=AsyncMock(return_value=job_id)):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/api/v1/invoices/upload",
                        files={"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")},
                        headers={"Authorization": "Bearer fake.jwt.token"},
                    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert "invoice_id" in data
    assert "job_id" in data


@pytest.mark.asyncio
async def test_list_invoices_returns_empty_for_new_tenant():
    mock_db = _make_mock_db()
    mock_queue = _make_mock_queue()

    # scalar for COUNT returns 0, execute returns empty list
    mock_db.scalar = AsyncMock(return_value=0)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute = AsyncMock(return_value=mock_result)

    app = _build_app_with_overrides(mock_db, mock_queue)

    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/v1/invoices",
                headers={"Authorization": "Bearer fake.jwt.token"},
            )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data"]["total"] == 0
    assert body["data"]["items"] == []


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf_exe():
    mock_db = _make_mock_db()
    mock_queue = _make_mock_queue()

    app = _build_app_with_overrides(mock_db, mock_queue)

    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/invoices/upload",
                files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
                headers={"Authorization": "Bearer fake.jwt.token"},
            )

    assert response.status_code == 400, response.text
