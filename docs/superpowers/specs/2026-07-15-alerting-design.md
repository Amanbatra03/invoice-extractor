# Invoice Extractor — Developer Alerting System Design

**Date:** 2026-07-15
**Status:** Approved — ready for implementation planning

---

## Problem

When a component of the system fails — a background job crashes, an API request hits an
unhandled exception, a tenant webhook exhausts its retries — the only trace today is a
structlog line on stdout and (for jobs) a `failed` row in `jobs`. Nobody is notified.
Developers should be alerted in near real-time, and every alert should be persisted for
later inspection.

## Requirements

- **Channel:** Discord webhook (single ops channel, URL supplied via env).
- **Scope:** all in-process failures:
  1. Worker job crashes (ingest, extract, batch, webhook delivery job, any future job).
  2. Unhandled API exceptions (500s).
  3. Tenant webhooks that permanently fail after all retries.
  Out of scope: service-down detection (dead processes can't alert about themselves;
  external uptime monitoring is a separate concern).
- **Persistence:** every alert stored in a new `alerts` table; admin-only API to browse.
- **Storm control:** repeated identical failures within a cooldown window must not spam
  Discord (but must still be recorded in the DB).
- **Safety:** the alerting path must never crash or block the component that is failing.

## Architecture

```
failure site ──► raise_alert() ──► INSERT alerts row (delivery_status=pending|suppressed)
                     │
                     ├─ Redis cooldown check (SET NX EX per fingerprint)
                     │      suppressed → stop (row kept, no Discord)
                     │
                     └─ enqueue workers.alert_job.run(alert_id) on "invoice-jobs"
                                   │
                                   └─ POST Discord embed (retries 5s/30s/120s)
                                          → delivered | failed (+last_error) | skipped
```

## Components

### 1. `Alert` model (`db/models.py`) + Alembic migration 0003

| Column | Type | Notes |
|---|---|---|
| `id` | UUID pk | `uuid.uuid4` default, matching existing models |
| `severity` | String(10), not null | `error` \| `warning` |
| `source` | String(50), not null, indexed | `worker` \| `api` \| `webhook_delivery` |
| `event` | String(100), not null | e.g. `job.failed:workers.extract_job.run`, `api.unhandled_exception`, `webhook.permanently_failed` |
| `detail` | Text, not null | exception message, truncated to 2000 chars |
| `context` | JSONB, nullable | job_id, invoice_id, method, path, request_id, tenant_id — whatever the hook knows |
| `fingerprint` | String(64), not null, indexed | sha256 of `source:event`, hex-truncated to 16 chars |
| `delivery_status` | String(20), not null, server_default `pending` | `pending` \| `delivered` \| `failed` \| `suppressed` \| `skipped` |
| `delivery_attempts` | Integer, not null, server_default `0` | |
| `last_error` | Text, nullable | last Discord delivery error |
| `delivered_at` | DateTime(tz), nullable | |
| `created_at` | DateTime(tz), not null, default `_now` | |

No `tenant_id` FK: alerts are operator-level and may be cross-tenant. Tenant, when
known, goes in `context`.

### 2. Alert service (`api/services/alerts.py`)

```python
async def raise_alert(
    db: AsyncSession, *,
    severity: str, source: str, event: str,
    detail: str, context: dict | None = None,
) -> Alert | None
```

Behavior (in this order):
1. Compute `fingerprint = sha256(f"{source}:{event}").hexdigest()[:16]`.
2. `ALERT_DISCORD_WEBHOOK_URL` empty → insert row with `delivery_status="skipped"`,
   never touch Redis or the queue, return.
3. Cooldown check against Redis: `SET alert:cd:{fingerprint} 1 NX EX {ALERT_COOLDOWN_SECONDS}`.
   - Key already present → insert row with `delivery_status="suppressed"`, return.
   - Key absent (set succeeded) → insert row `pending`, enqueue
     `workers.alert_job.run(str(alert.id))` on the existing `invoice-jobs` queue.
4. The entire body is wrapped in `try/except Exception`: on any internal failure
   (DB down, Redis down, enqueue failure) it logs `alert.raise_failed` via structlog and
   returns `None`. **`raise_alert` never raises.**
   - Redis unavailable → treat as *not* suppressed (fail open) and still attempt enqueue;
     if enqueue also fails, the row stays `pending` and the failure is logged.

Also exposes a sync wrapper `raise_alert_sync(...)` (creates its own session via
`get_session_factory()` and drives the coroutine with `asyncio.run`) for use from
synchronous contexts like the RQ exception handler.

### 3. Discord dispatcher (`workers/alert_job.py`)

`run(alert_id_str)` — same shape as `webhook_job.py`:
1. Load the `Alert` row; missing row → log warning, return.
2. Build one Discord embed: title `[{ENV}] {severity.upper()} — {event}`, color by
   severity (`error` red 0xE74C3C, `warning` amber 0xF39C12), fields for source, detail
   (truncated to 1000 chars), context rendered as `key: value` lines, ISO timestamp.
3. POST to `ALERT_DISCORD_WEBHOOK_URL` (httpx, timeout 10s), attempts at delays
   `[0, 5, 30, 120]`, incrementing `delivery_attempts` per attempt.
4. 2xx → `delivered` + `delivered_at`; all attempts exhausted → `failed` + `last_error`.

The dispatcher itself is a normal RQ job; if it raises, the worker exception handler
must **not** recurse — the handler skips alerting when `job.func_name` is
`workers.alert_job.run` (it logs instead).

### 4. Hook points

1. **Worker jobs** — custom RQ exception handler in `workers/worker.py`:
   ```python
   Worker(queues, connection=conn, exception_handlers=[alert_exception_handler])
   ```
   `alert_exception_handler(job, exc_type, exc_value, tb)` calls `raise_alert_sync` with
   `severity="error"`, `source="worker"`, `event=f"job.failed:{job.func_name}"`,
   `detail=str(exc_value)`, `context={"job_id": job.id, "args": [str(a) for a in job.args]}`,
   then returns `True` so RQ's default handling (FailedJobRegistry) still runs.
   Skips `workers.alert_job.run` to prevent recursion.
2. **API** — global handler registered in `create_app()`:
   `@app.exception_handler(Exception)` → `raise_alert` using a fresh session from
   `get_session_factory()` (the request-scoped session may be dead), `severity="error"`,
   `source="api"`, `event="api.unhandled_exception"`,
   `context={"method": ..., "path": ..., "request_id": ...}`; responds with the existing
   envelope: `{"data": None, "error": "Internal server error", "request_id": ...}`, 500.
3. **Webhook permanent failure** — in `webhook_job.py`'s `permanently_failed` branch:
   `raise_alert(db, severity="warning", source="webhook_delivery",
   event="webhook.permanently_failed", detail=delivery.last_error or "delivery failed",
   context={"webhook_id": ..., "event": ..., "attempts": ...})` reusing the job's open
   session.

### 5. Admin API (`api/routers/alerts.py`)

`GET /api/v1/alerts` — `require_roles("admin")`, mirrors the audit router:
query params `severity`, `source`, `limit` (default 50, max 200), `offset`;
ordered `created_at DESC`; returns `list[AlertOut]`
(`api/schemas/alert.py`: all columns, JSONB context passed through).

### 6. Configuration (`api/config.py`, `.env.example`)

```python
ALERT_DISCORD_WEBHOOK_URL: str = ""     # empty → alerts logged to DB only
ALERT_COOLDOWN_SECONDS: int = 600      # Discord suppression window per fingerprint
```

## Error handling summary

| Failure | Behavior |
|---|---|
| Redis down during cooldown check | fail open: attempt Discord enqueue anyway; log if that fails too |
| DB down during `raise_alert` | structlog `alert.raise_failed`; original failure path unaffected |
| Discord returns non-2xx / times out | retry 4 attempts total; then `failed` + `last_error` |
| `alert_job` itself crashes | worker handler logs but does not re-alert (no recursion) |
| No Discord URL configured | row saved as `skipped`; system is DB-log-only |

## Testing

TDD throughout; unit tests mock Redis/queue/httpx, integration tests use the existing
fixtures.

- **Unit — service:** fingerprint stability; suppression when cooldown key exists;
  `skipped` when URL empty; enqueue called with alert id; never raises when DB session
  or Redis explode (exception injected via mock).
- **Unit — dispatcher:** embed payload shape; success → `delivered`; non-2xx then 2xx →
  retry then `delivered`; all failures → `failed` with `last_error`; missing alert row
  no-ops.
- **Unit — worker handler:** fake job → `raise_alert_sync` called with expected event;
  returns True; `workers.alert_job.run` job → not alerted.
- **Integration — API:** route that raises → 500 envelope + one `alerts` row with
  `source="api"`; admin lists it via `GET /api/v1/alerts`; non-admin → 403; severity
  filter works.
- **Integration — webhook:** permanent delivery failure → `warning` alert row.

## Out of scope (YAGNI)

- Service-down / heartbeat detection, uptime monitoring.
- Additional channels (Slack, email) — the dispatcher is a single module; adding a
  channel later is a new function, not a redesign.
- Alert acknowledgement/resolution workflow, Streamlit alerts page.
