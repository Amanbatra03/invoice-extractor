from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_db():
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    return db


def _settings(url="https://discord.test/hook", cooldown=600):
    s = MagicMock()
    s.ALERT_DISCORD_WEBHOOK_URL = url
    s.ALERT_COOLDOWN_SECONDS = cooldown
    s.REDIS_URL = "redis://localhost:6379"
    return s


def test_fingerprint_stable():
    from api.services.alerts import _fingerprint
    assert _fingerprint("worker", "job.failed:x") == _fingerprint("worker", "job.failed:x")
    assert _fingerprint("worker", "e") != _fingerprint("api", "e")
    assert len(_fingerprint("a", "b")) == 16


@pytest.mark.asyncio
async def test_skipped_when_no_webhook_url():
    from api.services import alerts
    db = _mock_db()
    with patch.object(alerts, "get_settings", return_value=_settings(url="")), \
         patch.object(alerts, "Redis") as mock_redis, \
         patch.object(alerts, "get_queue") as mock_queue:
        alert = await alerts.raise_alert(
            db, severity="error", source="api", event="e", detail="boom"
        )
    assert alert.delivery_status == "skipped"
    mock_redis.from_url.assert_not_called()
    mock_queue.assert_not_called()
    db.add.assert_called_once()


@pytest.mark.asyncio
async def test_suppressed_when_cooldown_active():
    from api.services import alerts
    db = _mock_db()
    conn = MagicMock()
    conn.set.return_value = None  # SET NX failed -> key exists -> cooldown active
    with patch.object(alerts, "get_settings", return_value=_settings()), \
         patch.object(alerts, "Redis") as mock_redis, \
         patch.object(alerts, "get_queue") as mock_queue:
        mock_redis.from_url.return_value = conn
        alert = await alerts.raise_alert(
            db, severity="error", source="worker", event="job.failed:x", detail="boom"
        )
    assert alert.delivery_status == "suppressed"
    mock_queue.assert_not_called()


@pytest.mark.asyncio
async def test_pending_and_enqueued_when_not_suppressed():
    from api.services import alerts
    db = _mock_db()
    conn = MagicMock()
    conn.set.return_value = True  # SET NX succeeded
    queue = MagicMock()
    with patch.object(alerts, "get_settings", return_value=_settings()), \
         patch.object(alerts, "Redis") as mock_redis, \
         patch.object(alerts, "get_queue", return_value=queue):
        mock_redis.from_url.return_value = conn
        alert = await alerts.raise_alert(
            db, severity="error", source="worker", event="job.failed:x",
            detail="boom", context={"job_id": "j1"},
        )
    assert alert.delivery_status == "pending"
    queue.enqueue.assert_called_once_with("workers.alert_job.run", str(alert.id))
    conn.set.assert_called_once_with(
        f"alert:cd:{alert.fingerprint}", "1", nx=True, ex=600
    )


@pytest.mark.asyncio
async def test_fails_open_when_redis_down():
    from api.services import alerts
    db = _mock_db()
    queue = MagicMock()
    with patch.object(alerts, "get_settings", return_value=_settings()), \
         patch.object(alerts, "Redis") as mock_redis, \
         patch.object(alerts, "get_queue", return_value=queue):
        mock_redis.from_url.side_effect = ConnectionError("redis down")
        alert = await alerts.raise_alert(
            db, severity="error", source="api", event="e", detail="boom"
        )
    assert alert.delivery_status == "pending"
    queue.enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_never_raises_when_db_fails():
    from api.services import alerts
    db = _mock_db()
    db.commit = AsyncMock(side_effect=RuntimeError("db down"))
    with patch.object(alerts, "get_settings", return_value=_settings(url="")):
        alert = await alerts.raise_alert(
            db, severity="error", source="api", event="e", detail="boom"
        )
    assert alert is None  # swallowed, logged, no exception


@pytest.mark.asyncio
async def test_detail_truncated_to_2000_chars():
    from api.services import alerts
    db = _mock_db()
    with patch.object(alerts, "get_settings", return_value=_settings(url="")):
        alert = await alerts.raise_alert(
            db, severity="error", source="api", event="e", detail="x" * 5000
        )
    assert len(alert.detail) == 2000
