import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _session_factory(webhook):
    db = AsyncMock()
    db.scalar = AsyncMock(return_value=webhook)
    db.commit = AsyncMock()
    db.add = MagicMock()
    db.refresh = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=db)
    cm.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=cm)
    return factory, db


def _failing_httpx():
    client = AsyncMock()
    resp = MagicMock()
    resp.is_success = False
    resp.status_code = 500
    resp.text = "upstream error"
    client.post = AsyncMock(return_value=resp)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


@pytest.mark.asyncio
async def test_permanent_webhook_failure_raises_warning_alert():
    from workers import webhook_job

    webhook = MagicMock()
    webhook.secret = "s3cr3t"
    webhook.url = "https://customer.example/hook"
    factory, db = _session_factory(webhook)

    with patch.object(webhook_job, "get_session_factory", return_value=factory), \
         patch.object(webhook_job.httpx, "AsyncClient", _failing_httpx()), \
         patch.object(webhook_job.asyncio, "sleep", new=AsyncMock()), \
         patch.object(webhook_job, "raise_alert", new=AsyncMock()) as mock_alert:
        await webhook_job._run_async(str(uuid.uuid4()), "extraction.completed", {"event": "extraction.completed"})

    mock_alert.assert_awaited_once()
    kwargs = mock_alert.await_args.kwargs
    assert kwargs["severity"] == "warning"
    assert kwargs["source"] == "webhook_delivery"
    assert kwargs["event"] == "webhook.permanently_failed"
    assert "HTTP 500" in kwargs["detail"]
