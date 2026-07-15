import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from db.models import Alert


def _alert():
    return Alert(
        id=uuid.uuid4(),
        severity="error",
        source="worker",
        event="job.failed:workers.extract_job.run",
        detail="boom",
        context={"job_id": "j1"},
        fingerprint="abc123",
        delivery_status="pending",
        delivery_attempts=0,
        created_at=datetime.now(timezone.utc),
    )


def _session_factory(alert):
    db = AsyncMock()
    db.scalar = AsyncMock(return_value=alert)
    db.commit = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=db)
    cm.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=cm)
    return factory, db


def _httpx_client(responses):
    client = AsyncMock()
    client.post = AsyncMock(side_effect=responses)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm), client


def _settings(url="https://discord.test/hook"):
    s = MagicMock()
    s.ALERT_DISCORD_WEBHOOK_URL = url
    s.ENV = "test"
    return s


def _resp(ok=True, status=204):
    r = MagicMock()
    r.is_success = ok
    r.status_code = status
    r.text = "" if ok else "error body"
    return r


@pytest.mark.asyncio
async def test_delivered_on_first_success():
    from workers import alert_job
    alert = _alert()
    factory, db = _session_factory(alert)
    client_cls, client = _httpx_client([_resp(ok=True)])
    with patch.object(alert_job, "get_session_factory", return_value=factory), \
         patch.object(alert_job, "get_settings", return_value=_settings()), \
         patch.object(alert_job.httpx, "AsyncClient", client_cls):
        await alert_job._run_async(str(alert.id))
    assert alert.delivery_status == "delivered"
    assert alert.delivery_attempts == 1
    assert alert.delivered_at is not None
    payload = client.post.call_args.kwargs["json"]
    embed = payload["embeds"][0]
    assert embed["title"] == "[test] ERROR — job.failed:workers.extract_job.run"
    assert embed["color"] == 0xE74C3C


@pytest.mark.asyncio
async def test_retries_then_delivers():
    from workers import alert_job
    alert = _alert()
    factory, db = _session_factory(alert)
    client_cls, client = _httpx_client([_resp(ok=False, status=500), _resp(ok=True)])
    with patch.object(alert_job, "get_session_factory", return_value=factory), \
         patch.object(alert_job, "get_settings", return_value=_settings()), \
         patch.object(alert_job.httpx, "AsyncClient", client_cls), \
         patch.object(alert_job.asyncio, "sleep", new=AsyncMock()):
        await alert_job._run_async(str(alert.id))
    assert alert.delivery_status == "delivered"
    assert alert.delivery_attempts == 2


@pytest.mark.asyncio
async def test_failed_after_all_attempts():
    from workers import alert_job
    alert = _alert()
    factory, db = _session_factory(alert)
    client_cls, client = _httpx_client([_resp(ok=False, status=500)] * 4)
    with patch.object(alert_job, "get_session_factory", return_value=factory), \
         patch.object(alert_job, "get_settings", return_value=_settings()), \
         patch.object(alert_job.httpx, "AsyncClient", client_cls), \
         patch.object(alert_job.asyncio, "sleep", new=AsyncMock()):
        await alert_job._run_async(str(alert.id))
    assert alert.delivery_status == "failed"
    assert alert.delivery_attempts == 4
    assert "HTTP 500" in alert.last_error


@pytest.mark.asyncio
async def test_missing_alert_row_noops():
    from workers import alert_job
    factory, db = _session_factory(None)
    client_cls, client = _httpx_client([])
    with patch.object(alert_job, "get_session_factory", return_value=factory), \
         patch.object(alert_job, "get_settings", return_value=_settings()), \
         patch.object(alert_job.httpx, "AsyncClient", client_cls):
        await alert_job._run_async(str(uuid.uuid4()))
    client.post.assert_not_called()
