# Developer Alerting System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When any component fails (worker job crash, unhandled API 500, permanently-failed tenant webhook), persist an alert row and notify developers on Discord — without ever letting the alerting path crash the thing it monitors.

**Architecture:** A central `raise_alert()` service writes an `alerts` row, applies a Redis cooldown per failure fingerprint, and enqueues an RQ job that posts a Discord embed with retries. Three hook points: an RQ exception handler (all worker jobs), a global FastAPI exception handler (all API 500s), and the webhook permanent-failure branch. Admin-only `GET /api/v1/alerts` browses history.

**Tech Stack:** FastAPI, SQLAlchemy async + Alembic, RQ/Redis, httpx, structlog, pytest (existing suite conventions: mocked AsyncSession, `ASGITransport` integration tests).

**Spec:** `docs/superpowers/specs/2026-07-15-alerting-design.md`

**Environment note:** Run all pytest commands with the Anaconda base interpreter:
`C:\Users\amanb\anaconda3\python.exe -m pytest ...` from the repo root
(`C:\Users\amanb\invoice-extractor`). Bare `python` on this machine is a bare
Python 3.14 without project dependencies.

---

## File Structure

```
api/config.py                          MODIFY — 2 new settings
.env.example                           MODIFY — document new settings
db/models.py                           MODIFY — add Alert model
db/migrations/versions/0003_alerts.py  CREATE — alerts table
api/services/alerts.py                 CREATE — raise_alert / raise_alert_sync (the only entry point)
workers/alert_job.py                   CREATE — Discord embed dispatcher (RQ job)
workers/worker.py                      MODIFY — register exception handler
api/main.py                            MODIFY — global exception handler + alerts router
workers/webhook_job.py                 MODIFY — alert on permanent failure
api/schemas/alert.py                   CREATE — AlertOut
api/routers/alerts.py                  CREATE — admin listing endpoint

tests/unit/test_config_alerts.py       CREATE
tests/unit/test_alert_model.py         CREATE
tests/unit/test_alerts_service.py      CREATE
tests/unit/test_alert_job.py           CREATE
tests/unit/test_worker_alert_handler.py CREATE
tests/unit/test_webhook_alert.py       CREATE
tests/integration/test_alerts_api.py   CREATE
```

---

### Task 1: Config settings

**Files:**
- Modify: `api/config.py` (after the `SENTRY_DSN` block, line ~42)
- Modify: `.env.example`
- Test: `tests/unit/test_config_alerts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_config_alerts.py`:

```python
import os
from unittest.mock import patch

_TEST_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
    "REDIS_URL": "redis://localhost:6379",
    "SUPABASE_URL": "https://fake.supabase.co",
    "SUPABASE_ANON_KEY": "fake_anon",
    "SUPABASE_SERVICE_KEY": "fake_service",
    "SUPABASE_JWT_SECRET": "test_secret",
    "GOOGLE_API_KEY": "fake_google_key",
}


def test_alert_settings_defaults():
    from api.config import get_settings
    with patch.dict(os.environ, _TEST_ENV):
        get_settings.cache_clear()
        s = get_settings()
        assert s.ALERT_DISCORD_WEBHOOK_URL == ""
        assert s.ALERT_COOLDOWN_SECONDS == 600
    get_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_config_alerts.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'ALERT_DISCORD_WEBHOOK_URL'`

- [ ] **Step 3: Add the settings**

In `api/config.py`, after the `SENTRY_DSN: str = ""` line, add:

```python
    # Developer alerting
    ALERT_DISCORD_WEBHOOK_URL: str = ""   # empty -> alerts logged to DB only
    ALERT_COOLDOWN_SECONDS: int = 600     # Discord suppression window per fingerprint
```

In `.env.example`, append:

```
# Developer alerting (Discord webhook; leave empty to log alerts to DB only)
ALERT_DISCORD_WEBHOOK_URL=
ALERT_COOLDOWN_SECONDS=600
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_config_alerts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/config.py .env.example tests/unit/test_config_alerts.py
git commit -m "feat(alerts): add alerting settings to config"
```

---

### Task 2: Alert model + Alembic migration 0003

**Files:**
- Modify: `db/models.py` (append after `AuditLog`, before `Conversation`)
- Create: `db/migrations/versions/0003_alerts.py`
- Test: `tests/unit/test_alert_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_alert_model.py`:

```python
def test_alert_model_columns():
    from db.models import Alert
    cols = {c.name for c in Alert.__table__.columns}
    expected = {
        "id", "severity", "source", "event", "detail", "context",
        "fingerprint", "delivery_status", "delivery_attempts",
        "last_error", "delivered_at", "created_at",
    }
    assert expected <= cols
    assert Alert.__tablename__ == "alerts"


def test_alert_model_has_no_tenant_fk():
    from db.models import Alert
    assert "tenant_id" not in {c.name for c in Alert.__table__.columns}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_alert_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'Alert'`

- [ ] **Step 3: Add the model**

In `db/models.py`, after the `AuditLog` class, add:

```python
class Alert(Base):
    __tablename__ = "alerts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    severity = Column(String(10), nullable=False)
    source = Column(String(50), nullable=False, index=True)
    event = Column(String(100), nullable=False)
    detail = Column(Text, nullable=False)
    context = Column(JSONB, nullable=True)
    fingerprint = Column(String(64), nullable=False, index=True)
    delivery_status = Column(String(20), nullable=False, server_default="pending")
    delivery_attempts = Column(Integer, nullable=False, server_default="0")
    last_error = Column(Text, nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
```

- [ ] **Step 4: Create the migration**

Create `db/migrations/versions/0003_alerts.py`:

```python
"""alerts table for developer alerting

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-15
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("severity", sa.String(10), nullable=False),
        sa.Column("source", sa.String(50), nullable=False, index=True),
        sa.Column("event", sa.String(100), nullable=False),
        sa.Column("detail", sa.Text, nullable=False),
        sa.Column("context", postgresql.JSONB, nullable=True),
        sa.Column("fingerprint", sa.String(64), nullable=False, index=True),
        sa.Column("delivery_status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("delivery_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text, nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("alerts")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_alert_model.py tests/unit/test_models.py -v`
Expected: PASS (including the existing model tests — no regressions)

- [ ] **Step 6: Commit**

```bash
git add db/models.py db/migrations/versions/0003_alerts.py tests/unit/test_alert_model.py
git commit -m "feat(alerts): add Alert model and migration 0003"
```

---

### Task 3: Alert service — `raise_alert` / `raise_alert_sync`

**Files:**
- Create: `api/services/alerts.py`
- Test: `tests/unit/test_alerts_service.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_alerts_service.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_alerts_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.services.alerts'`

- [ ] **Step 3: Implement the service**

Create `api/services/alerts.py`:

```python
import asyncio
import hashlib

import structlog
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import get_settings
from api.dependencies import get_queue
from db.models import Alert

log = structlog.get_logger()

_DETAIL_MAX = 2000


def _fingerprint(source: str, event: str) -> str:
    return hashlib.sha256(f"{source}:{event}".encode()).hexdigest()[:16]


async def raise_alert(
    db: AsyncSession,
    *,
    severity: str,
    source: str,
    event: str,
    detail: str,
    context: dict | None = None,
) -> Alert | None:
    """Persist an alert and (unless suppressed) enqueue Discord dispatch.

    Never raises: an alerting failure must not take down the failing
    component that is being reported.
    """
    try:
        settings = get_settings()
        fp = _fingerprint(source, event)
        alert = Alert(
            severity=severity,
            source=source,
            event=event,
            detail=(detail or "")[:_DETAIL_MAX],
            context=context,
            fingerprint=fp,
        )

        if not settings.ALERT_DISCORD_WEBHOOK_URL:
            alert.delivery_status = "skipped"
            db.add(alert)
            await db.commit()
            return alert

        suppressed = False
        try:
            conn = Redis.from_url(settings.REDIS_URL)
            acquired = conn.set(
                f"alert:cd:{fp}", "1", nx=True, ex=settings.ALERT_COOLDOWN_SECONDS
            )
            suppressed = not acquired
        except Exception as exc:
            # Fail open: no cooldown info means we'd rather alert than stay silent.
            log.warning("alert.cooldown_check_failed", error=str(exc))

        alert.delivery_status = "suppressed" if suppressed else "pending"
        db.add(alert)
        await db.commit()

        if not suppressed:
            try:
                get_queue().enqueue("workers.alert_job.run", str(alert.id))
            except Exception as exc:
                log.error("alert.enqueue_failed", alert_id=str(alert.id), error=str(exc))
        return alert

    except Exception as exc:
        log.error("alert.raise_failed", error=str(exc), source=source, event=event)
        return None


def raise_alert_sync(
    *,
    severity: str,
    source: str,
    event: str,
    detail: str,
    context: dict | None = None,
) -> None:
    """Wrapper for synchronous contexts (RQ exception handler)."""

    async def _run() -> None:
        from db.session import get_session_factory
        async with get_session_factory()() as db:
            await raise_alert(
                db, severity=severity, source=source, event=event,
                detail=detail, context=context,
            )

    try:
        asyncio.run(_run())
    except Exception as exc:
        log.error("alert.raise_sync_failed", error=str(exc), source=source, event=event)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_alerts_service.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add api/services/alerts.py tests/unit/test_alerts_service.py
git commit -m "feat(alerts): add raise_alert service with cooldown and never-raise guarantee"
```

---

### Task 4: Discord dispatcher — `workers/alert_job.py`

**Files:**
- Create: `workers/alert_job.py`
- Test: `tests/unit/test_alert_job.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_alert_job.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_alert_job.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workers.alert_job'`

- [ ] **Step 3: Implement the dispatcher**

Create `workers/alert_job.py`:

```python
import asyncio
import uuid
from datetime import datetime, timezone

import httpx
import structlog
from sqlalchemy import select

from api.config import get_settings
from db.models import Alert
from db.session import get_session_factory

log = structlog.get_logger()

_RETRY_DELAYS = [5, 30, 120]
_MAX_ATTEMPTS = len(_RETRY_DELAYS) + 1
_SEVERITY_COLORS = {"error": 0xE74C3C, "warning": 0xF39C12}


def _build_embed(alert: Alert, env: str) -> dict:
    fields = [
        {"name": "Source", "value": alert.source, "inline": True},
        {"name": "Severity", "value": alert.severity, "inline": True},
        {"name": "Detail", "value": (alert.detail or "-")[:1000], "inline": False},
    ]
    context_lines = "\n".join(f"**{k}**: {v}" for k, v in (alert.context or {}).items())
    if context_lines:
        fields.append({"name": "Context", "value": context_lines[:1000], "inline": False})
    created = alert.created_at or datetime.now(timezone.utc)
    return {
        "embeds": [{
            "title": f"[{env}] {alert.severity.upper()} — {alert.event}"[:256],
            "color": _SEVERITY_COLORS.get(alert.severity, 0x95A5A6),
            "fields": fields,
            "timestamp": created.isoformat(),
        }]
    }


def run(alert_id_str: str) -> None:
    asyncio.run(_run_async(alert_id_str))


async def _run_async(alert_id_str: str) -> None:
    settings = get_settings()
    alert_id = uuid.UUID(alert_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        alert = await db.scalar(select(Alert).where(Alert.id == alert_id))
        if not alert:
            log.warning("alert.not_found", alert_id=alert_id_str)
            return
        if not settings.ALERT_DISCORD_WEBHOOK_URL:
            alert.delivery_status = "skipped"
            await db.commit()
            return

        payload = _build_embed(alert, settings.ENV)

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if attempt > 0:
                await asyncio.sleep(delay)
            alert.delivery_attempts = attempt + 1
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        settings.ALERT_DISCORD_WEBHOOK_URL, json=payload
                    )
                if resp.is_success:
                    alert.delivery_status = "delivered"
                    alert.delivered_at = datetime.now(timezone.utc)
                    await db.commit()
                    log.info("alert.delivered", alert_id=alert_id_str, attempt=attempt + 1)
                    return
                alert.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                await db.commit()
                log.warning("alert.delivery_retry", attempt=attempt + 1, status=resp.status_code)
            except Exception as exc:
                alert.last_error = str(exc)[:500]
                await db.commit()
                log.warning("alert.delivery_error", attempt=attempt + 1, error=str(exc))

        alert.delivery_status = "failed"
        await db.commit()
        log.error("alert.delivery_failed", alert_id=alert_id_str)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_alert_job.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add workers/alert_job.py tests/unit/test_alert_job.py
git commit -m "feat(alerts): add Discord alert dispatcher job with retries"
```

---

### Task 5: RQ exception handler — all worker job failures

**Files:**
- Modify: `workers/worker.py` (whole file shown below)
- Test: `tests/unit/test_worker_alert_handler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_worker_alert_handler.py`:

```python
from unittest.mock import MagicMock, patch


def _job(func_name="workers.extract_job.run"):
    job = MagicMock()
    job.func_name = func_name
    job.id = "rq-job-1"
    job.args = ("inv-1", "job-1")
    return job


def test_handler_alerts_on_job_failure():
    from workers import worker
    job = _job()
    with patch.object(worker, "raise_alert_sync") as mock_alert:
        result = worker.alert_exception_handler(job, RuntimeError, RuntimeError("boom"), None)
    assert result is True  # RQ default handling (FailedJobRegistry) still runs
    mock_alert.assert_called_once_with(
        severity="error",
        source="worker",
        event="job.failed:workers.extract_job.run",
        detail="boom",
        context={"job_id": "rq-job-1", "args": ["inv-1", "job-1"]},
    )


def test_handler_skips_alert_job_to_avoid_recursion():
    from workers import worker
    job = _job(func_name="workers.alert_job.run")
    with patch.object(worker, "raise_alert_sync") as mock_alert:
        result = worker.alert_exception_handler(job, RuntimeError, RuntimeError("boom"), None)
    assert result is True
    mock_alert.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_worker_alert_handler.py -v`
Expected: FAIL with `AttributeError: module 'workers.worker' has no attribute 'alert_exception_handler'`

- [ ] **Step 3: Implement the handler**

Replace the full contents of `workers/worker.py` with:

```python
import structlog
from redis import Redis
from rq import Worker, Queue

from api.config import get_settings
from api.services.alerts import raise_alert_sync

log = structlog.get_logger()


def alert_exception_handler(job, exc_type, exc_value, tb) -> bool:
    """Fire a developer alert for any failed RQ job.

    Returning True lets RQ's default handling (FailedJobRegistry) run as before.
    The alert dispatcher itself is excluded to prevent alert-about-alert recursion.
    """
    if job.func_name == "workers.alert_job.run":
        log.error("alert.dispatcher_crashed", job_id=job.id, error=str(exc_value))
        return True
    raise_alert_sync(
        severity="error",
        source="worker",
        event=f"job.failed:{job.func_name}",
        detail=str(exc_value),
        context={"job_id": job.id, "args": [str(a) for a in job.args]},
    )
    return True


def main():
    settings = get_settings()
    conn = Redis.from_url(settings.REDIS_URL)
    queues = [Queue("invoice-jobs", connection=conn)]
    worker = Worker(queues, connection=conn, exception_handlers=[alert_exception_handler])
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_worker_alert_handler.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add workers/worker.py tests/unit/test_worker_alert_handler.py
git commit -m "feat(alerts): alert on any RQ job failure via worker exception handler"
```

---

### Task 6: Global API exception handler

**Files:**
- Modify: `api/main.py`
- Test: `tests/integration/test_alerts_api.py` (exception-handler tests; the router tests are Task 8)

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_alerts_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/integration/test_alerts_api.py -v`
Expected: FAIL — `AttributeError: <module 'api.main'> does not have the attribute 'raise_alert'` (and/or 500s bubbling without the envelope)

- [ ] **Step 3: Implement the handler**

In `api/main.py`:

1. Add to the import block at the top:

```python
from starlette.requests import Request

from api.services.alerts import raise_alert
from db.session import get_session_factory
```

2. Add after the `structlog.configure(...)` call:

```python
log = structlog.get_logger()
```

3. Inside `create_app()`, right after the `RateLimitExceeded` handler registration, add:

```python
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        try:
            async with get_session_factory()() as db:
                await raise_alert(
                    db,
                    severity="error",
                    source="api",
                    event="api.unhandled_exception",
                    detail=str(exc),
                    context={
                        "method": request.method,
                        "path": request.url.path,
                        "request_id": request_id,
                    },
                )
        except Exception as alert_exc:
            log.error("alert.handler_failed", error=str(alert_exc))
        return JSONResponse(
            status_code=500,
            content={"data": None, "error": "Internal server error", "request_id": request_id},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/integration/test_alerts_api.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run the full integration suite for regressions**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/integration -m "not slow" -q`
Expected: all PASS (the new 500 handler must not swallow existing HTTPException flows — Starlette routes those to their own handler, not this one)

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/integration/test_alerts_api.py
git commit -m "feat(alerts): alert on unhandled API exceptions via global handler"
```

---

### Task 7: Webhook permanent-failure alert

**Files:**
- Modify: `workers/webhook_job.py`
- Test: `tests/unit/test_webhook_alert.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_webhook_alert.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_webhook_alert.py -v`
Expected: FAIL with `AttributeError: <module 'workers.webhook_job'> does not have the attribute 'raise_alert'`

- [ ] **Step 3: Wire in the alert**

In `workers/webhook_job.py`:

1. Add to the imports:

```python
from api.services.alerts import raise_alert
```

2. Replace the permanent-failure block at the end of `_run_async` (currently):

```python
            if attempt + 1 >= _MAX_ATTEMPTS:
                delivery.status = "failed"
                await db.commit()
                log.error("webhook.permanently_failed", webhook_id=webhook_id_str, event=event)
```

with:

```python
            if attempt + 1 >= _MAX_ATTEMPTS:
                delivery.status = "failed"
                await db.commit()
                log.error("webhook.permanently_failed", webhook_id=webhook_id_str, event=event)
                await raise_alert(
                    db,
                    severity="warning",
                    source="webhook_delivery",
                    event="webhook.permanently_failed",
                    detail=delivery.last_error or "delivery failed",
                    context={
                        "webhook_id": webhook_id_str,
                        "event": event,
                        "attempts": delivery.attempts,
                    },
                )
```

- [ ] **Step 4: Run tests to verify they pass (including existing webhook tests)**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/unit/test_webhook_alert.py tests/unit -k "webhook" -v`
Expected: PASS, no regressions

- [ ] **Step 5: Commit**

```bash
git add workers/webhook_job.py tests/unit/test_webhook_alert.py
git commit -m "feat(alerts): warning alert when tenant webhook permanently fails"
```

---

### Task 8: Admin alerts endpoint

**Files:**
- Create: `api/schemas/alert.py`
- Create: `api/routers/alerts.py`
- Modify: `api/main.py` (router registration)
- Test: `tests/integration/test_alerts_api.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/test_alerts_api.py`:

```python
def _alert_row():
    import datetime
    from db.models import Alert
    return Alert(
        id=uuid.uuid4(),
        severity="error",
        source="api",
        event="api.unhandled_exception",
        detail="kaboom",
        context={"path": "/api/v1/x"},
        fingerprint="abc123",
        delivery_status="delivered",
        delivery_attempts=1,
        last_error=None,
        delivered_at=datetime.datetime.now(datetime.timezone.utc),
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )


@pytest.mark.asyncio
async def test_admin_lists_alerts():
    mock_db = _make_mock_db()
    row = _alert_row()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [row]
    mock_db.execute = AsyncMock(return_value=mock_result)

    with patch.dict(os.environ, _TEST_ENV):
        app = _build_app(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=ADMIN_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/alerts",
                    headers={"Authorization": "Bearer faketoken"},
                )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    assert len(body["data"]) == 1
    assert body["data"][0]["event"] == "api.unhandled_exception"
    assert body["data"][0]["delivery_status"] == "delivered"


@pytest.mark.asyncio
async def test_non_admin_gets_403():
    mock_db = _make_mock_db()
    with patch.dict(os.environ, _TEST_ENV):
        app = _build_app(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=VIEWER_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/alerts",
                    headers={"Authorization": "Bearer faketoken"},
                )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_severity_filter_accepted():
    mock_db = _make_mock_db()
    with patch.dict(os.environ, _TEST_ENV):
        app = _build_app(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=ADMIN_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/alerts?severity=warning&source=worker&limit=10&offset=0",
                    headers={"Authorization": "Bearer faketoken"},
                )
    assert resp.status_code == 200
    assert resp.json()["data"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/integration/test_alerts_api.py -v`
Expected: new tests FAIL with 404 (route not registered)

- [ ] **Step 3: Create the schema**

Create `api/schemas/alert.py`:

```python
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    severity: str
    source: str
    event: str
    detail: str
    context: dict | None
    fingerprint: str
    delivery_status: str
    delivery_attempts: int
    last_error: str | None
    delivered_at: datetime | None
    created_at: datetime
```

- [ ] **Step 4: Create the router**

Create `api/routers/alerts.py`:

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from api.schemas.alert import AlertOut
from db.models import Alert
from db.session import get_db

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=dict)
async def list_alerts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    severity: str | None = Query(None),
    source: str | None = Query(None),
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    q = select(Alert).order_by(Alert.created_at.desc())
    if severity:
        q = q.where(Alert.severity == severity)
    if source:
        q = q.where(Alert.source == source)
    rows = (await db.execute(q.limit(limit).offset(offset))).scalars().all()
    return {
        "data": [AlertOut.model_validate(r).model_dump(mode="json") for r in rows],
        "error": None,
        "request_id": None,
    }
```

- [ ] **Step 5: Register the router**

In `api/main.py`, add to the router import block:

```python
from api.routers import alerts as alerts_router
```

and after the `chat_router` include line:

```python
    app.include_router(alerts_router.router, prefix="/api/v1")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest tests/integration/test_alerts_api.py -v`
Expected: 5 PASS (2 from Task 6 + 3 new)

- [ ] **Step 7: Commit**

```bash
git add api/schemas/alert.py api/routers/alerts.py api/main.py tests/integration/test_alerts_api.py
git commit -m "feat(alerts): admin-only alerts listing endpoint"
```

---

### Task 9: Full-suite verification, docs, push

**Files:**
- Modify: `README.md` (Alerting subsection under the enterprise features)

- [ ] **Step 1: Run the entire fast suite**

Run: `C:\Users\amanb\anaconda3\python.exe -m pytest -m "not slow" -q`
Expected: all PASS (176 pre-existing + ~19 new), no warnings introduced by alert modules

- [ ] **Step 2: Document the feature**

In `README.md`, add under the enterprise features section:

```markdown
### Developer Alerting

Failures anywhere in the stack raise a developer alert: worker job crashes
(RQ exception handler), unhandled API 500s (global exception handler), and
tenant webhooks that exhaust their retries. Alerts are persisted to the
`alerts` table and delivered to Discord as embeds with automatic retries and
a per-fingerprint cooldown (default 10 min) to prevent alert storms.

- Configure: set `ALERT_DISCORD_WEBHOOK_URL` (leave empty for DB-log-only mode)
- Browse: `GET /api/v1/alerts` (admin role) with `severity`/`source` filters
- Migration: `alembic upgrade head` (adds the `alerts` table, revision 0003)
```

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: document developer alerting system"
git push origin enterprise-upgrade
```

---

## Self-Review Notes

- Spec coverage: model/migration (Task 2), service with ordering URL-check → cooldown → enqueue (Task 3), dispatcher with skipped/delivered/failed transitions (Task 4), worker hook with recursion guard (Task 5), API hook using a fresh session + envelope (Task 6), webhook hook reusing the job's session (Task 7), admin endpoint + schema (Task 8), config (Task 1). Out-of-scope items (heartbeat, extra channels, ack workflow) have no tasks — intentional.
- The `suppressed` path never enqueues, but the row is still written (Task 3 test `test_suppressed_when_cooldown_active`).
- `raise_alert_sync` signature in Task 3 matches its use in Task 5 (`severity/source/event/detail/context` kwargs, no `db`).
- `workers.alert_job.run` string enqueued in Task 3 matches the module created in Task 4.
- Discord delivery attempts: `[0] + [5, 30, 120]` = 4 total, matching spec's "retry 4 attempts total" and Task 4's `test_failed_after_all_attempts`.
