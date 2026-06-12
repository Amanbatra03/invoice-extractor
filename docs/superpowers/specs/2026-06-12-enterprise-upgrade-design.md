# Invoice Extractor — Enterprise Production Design

**Date:** 2026-06-12  
**Author:** Aman Batra + Claude Code  
**Status:** Approved — ready for implementation planning

---

## 1. Problem Statement

The current invoice-extractor is a Streamlit monolith with local filesystem storage, single-password auth, and no API layer. It cannot support multiple users, horizontal scaling, ERP integrations, or enterprise security requirements. This document defines the architecture to make it enterprise production-ready.

---

## 2. Requirements

### Functional
- **Three workflows:** Interactive analyst UI, high-volume batch processing, API-first ERP integration
- **Full RBAC:** `admin`, `analyst`, `viewer`, `api_user` roles with enforced permissions
- **Multi-tenancy:** Complete tenant isolation at the database layer via Supabase RLS
- **Webhook integrations:** Push events to ERP/accounting systems on extraction completion
- **Batch processing:** Upload and extract 50–500 invoices asynchronously with progress polling
- **Audit trail:** Append-only log of every write action for compliance

### Non-Functional
- **Deployment:** Cloud-native SaaS on Render (Docker)
- **Storage:** Supabase (Postgres + pgvector + Storage) — no local filesystem in production
- **LLM:** Gemini 2.0 Flash (production) + Gemma 3 4B via Ollama (local dev fallback)
- **Embeddings:** `gemini-embedding-exp-03-07` (768 dims) in production, Gemma 3 4B locally
- **Observability:** structlog (JSON), LangSmith (LLM traces), Sentry (errors), Prometheus (metrics)

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENTS                                  │
│  Streamlit Analyst UI   │   ERP / 3rd-party (REST + Webhooks)  │
└────────────┬────────────┴──────────────┬────────────────────────┘
             │  HTTP + JWT               │  API Key + JWT
             ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend  (Render Web)                 │
│  /api/v1/invoices   /api/v1/extract   /api/v1/qa               │
│  /api/v1/compare    /api/v1/batch     /api/v1/webhooks          │
│  /api/v1/users      /api/v1/jobs      /api/v1/audit             │
│                                                                 │
│  Auth middleware (Supabase JWT)  │  RBAC dependency injection   │
│  Rate limiter (per-tenant)       │  Request ID tracing          │
│  File validation                 │  Structured logging          │
└──────┬──────────────┬────────────┴──────────────────────────────┘
       │ enqueue      │ DB/vector/files
       ▼              ▼
┌─────────────┐  ┌────────────────────────────────────────────────┐
│  Redis      │  │              Supabase                          │
│  (job queue)│  │  Postgres: tenants, users, invoices,           │
└──────┬──────┘  │            extractions, jobs, audit_log,       │
       │         │            webhooks, api_keys, llm_usage       │
       ▼         │  pgvector:  invoice_chunks (replaces ChromaDB) │
┌─────────────┐  │  Storage:   PDFs + images (per-tenant bucket)  │
│  RQ Worker  │  │  Auth:      JWT + RLS tenant isolation         │
│  (Render    │  └────────────────────────────────────────────────┘
│   Worker)   │
│  • ingest   │  ┌────────────────────────────────────────────────┐
│  • extract  ├─►│           AI / LLM Layer                       │
│  • batch    │  │  Gemini 2.0 Flash  (text + vision, prod)       │
│  • webhook  │  │  gemini-embedding-exp-03-07 (768 dims, prod)   │
│    dispatch │  │  Gemma 3 4B via Ollama (local dev fallback)    │
└─────────────┘  │  LLMProvider protocol (config-driven switch)   │
                 └────────────────────────────────────────────────┘
```

**Key structural decisions:**
- FastAPI is the spine — Streamlit and ERP systems are both clients. Zero business logic in Streamlit.
- pgvector replaces ChromaDB — embeddings in `invoice_chunks` table with tenant RLS. BM25 rebuilt in-memory per request.
- Supabase Storage replaces local `data/` dirs — PDFs/images in per-tenant buckets, signed URLs for access.
- RQ Worker on Render handles all slow work so the API stays fast and responsive.
- `LLM_PROVIDER=gemini` in production, `LLM_PROVIDER=ollama_gemma` in local dev — single config switch.

---

## 4. Data Model

```sql
-- Every table has tenant_id. RLS: tenant_id = auth.jwt() ->> 'tenant_id'

tenants (id uuid PK, name text, plan text DEFAULT 'free', created_at timestamptz)

users (id uuid PK, tenant_id uuid FK, email text,
       role text CHECK IN ('admin','analyst','viewer','api_user'), created_at timestamptz)

api_keys (id uuid PK, tenant_id uuid FK, name text, key_hash text,
          role text DEFAULT 'api_user', last_used_at timestamptz, created_at timestamptz)

invoices (id uuid PK, tenant_id uuid FK, uploaded_by uuid FK → users,
          file_name text, file_type text CHECK IN ('pdf','image'),
          storage_path text, sha256 text,
          status text CHECK IN ('pending','ingesting','ready','failed'),
          created_at timestamptz)

invoice_chunks (id uuid PK, invoice_id uuid FK, tenant_id uuid FK,
                chunk_text text, page_num int,
                embedding vector(768),           -- gemini-embedding-exp-03-07
                created_at timestamptz)

-- pgvector index:
-- CREATE INDEX ON invoice_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

extractions (id uuid PK, invoice_id uuid FK, tenant_id uuid FK,
             schema_json jsonb, model_used text, validated bool DEFAULT false,
             created_at timestamptz)

jobs (id uuid PK, tenant_id uuid FK,
      type text CHECK IN ('ingest','extract','batch_extract','compare'),
      status text CHECK IN ('queued','running','done','failed'),
      payload jsonb, result jsonb, error text,
      created_at timestamptz, completed_at timestamptz)

webhooks (id uuid PK, tenant_id uuid FK, url text, events text[],
          secret text, active bool DEFAULT true, created_at timestamptz)

webhook_deliveries (id uuid PK, webhook_id uuid FK, event text, payload jsonb,
                    status text CHECK IN ('pending','delivered','failed'),
                    attempts int DEFAULT 0, last_error text, delivered_at timestamptz)

audit_log (id uuid PK, tenant_id uuid FK, user_id uuid,
           action text, resource_type text, resource_id uuid,
           metadata jsonb, created_at timestamptz)

llm_usage (id uuid PK, tenant_id uuid FK, invoice_id uuid,
           model text, agent text, input_tokens int, output_tokens int,
           latency_ms int, cost_usd numeric(10,6), created_at timestamptz)
```

**Design notes:**
- `invoices.sha256` — dedup: same file uploaded twice returns existing invoice id, skips re-ingestion.
- `webhook_deliveries` — separate from job queue; tracks every retry attempt for admin visibility.
- `audit_log` — append-only enforced by RLS (INSERT only, no UPDATE/DELETE for any role).
- `api_keys.key_hash` — bcrypt only; raw key shown once at creation and never stored.

---

## 5. API Design

**Base:** `https://invoice-analyst.onrender.com/api/v1`  
**Auth:** `Authorization: Bearer <jwt>` (users) | `X-API-Key: <key>` (api_user)  
**Envelope:** `{ data, error, request_id }` on all responses

### Invoices
```
POST   /invoices/upload              Upload PDF or image → enqueues ingest job
GET    /invoices                     List (paginated, filterable by status/type)
GET    /invoices/{id}                Get metadata + extraction if ready
DELETE /invoices/{id}                Delete all assets (admin only)
GET    /invoices/{id}/download       Signed URL (15 min expiry)
```

### Extraction
```
POST   /invoices/{id}/extract        Enqueue extraction job
GET    /invoices/{id}/extraction     Get cached InvoiceSchema result
POST   /invoices/{id}/validate       Re-run arithmetic + anomaly checks
```

### Q&A
```
POST   /invoices/{id}/qa             Ask question → { answer, chunks, agent_trace }
POST   /invoices/{id}/qa/stream      Same, returns text/event-stream (SSE)
```

### Comparison
```
POST   /compare                      Compare 2+ invoices → { table, discrepancies }
```

### Batch
```
POST   /batch/extract                Enqueue batch extraction for N invoices
GET    /batch/{id}                   Poll status + per-invoice results
GET    /batch/{id}/export            Download all extractions as CSV
```

### Jobs
```
GET    /jobs/{id}                    Poll any single job
GET    /jobs                         List recent jobs for tenant
```

### Webhooks
```
POST   /webhooks                     Register endpoint
GET    /webhooks                     List all
GET    /webhooks/{id}                Get + delivery history
PATCH  /webhooks/{id}                Update url/events/active
DELETE /webhooks/{id}                Disable
POST   /webhooks/{id}/test           Send test ping
```

### Users & Keys
```
GET    /users                        List tenant users
PATCH  /users/{id}/role              Change role (admin only)
DELETE /users/{id}                   Remove user
POST   /api-keys                     Create key → raw shown once
GET    /api-keys                     List (hashes only)
DELETE /api-keys/{id}                Revoke
```

### Audit & Health
```
GET    /audit                        Paginated audit log (admin only)
GET    /health                       { status, db, redis }
GET    /health/ready                 Render readiness probe
GET    /metrics                      Prometheus (internal only)
GET    /usage                        Per-tenant LLM cost report (admin only)
```

### RBAC Matrix
```
Endpoint group        admin   analyst   viewer   api_user
Upload invoice          ✓       ✓         ✗        ✓
List / view invoices    ✓       ✓         ✓        ✓
Delete invoice          ✓       ✗         ✗        ✗
Run extraction          ✓       ✓         ✗        ✓
View extraction         ✓       ✓         ✓        ✓
Q&A                     ✓       ✓         ✓        ✗
Compare                 ✓       ✓         ✗        ✓
Batch extract           ✓       ✓         ✗        ✓
Manage webhooks         ✓       ✗         ✗        ✓
Manage users/keys       ✓       ✗         ✗        ✗
Audit log               ✓       ✗         ✗        ✗
```

---

## 6. AI Agents

Six agents — three upgraded from existing code, three new.

### Agent 1 — RAG Q&A Agent (upgraded)
LangGraph 5-node graph. Changes: ChromaDB → pgvector retrieval, Ollama → Gemini, SSE streaming added.
```
query_rewriter → pgvector_retrieve (+ BM25 RRF fusion) → relevance_grade
    ↑ retry if irrelevant                                        │
    └─────────────────────────────────────────── generate_answer → self_critique
```

### Agent 2 — Smart Extraction Agent (upgraded)
LangGraph graph. Replaces single-function extractor. Uses Gemini native structured output (`response_schema=InvoiceSchema`). Retry node feeds validation error back to Gemini for self-correction (max 2 retries).
```
all_chunks → build_context → gemini_structured_extract → pydantic_validate → done
                                        │ schema error
                                        └──► retry_with_error_hint
```

### Agent 3 — Batch Orchestrator Agent (new)
LangGraph with `Send` API for parallel fan-out. Processes N invoices concurrently. Writes per-invoice progress to `jobs` table mid-run for polling. Fires webhook on completion.
```
receive_batch → fan_out (parallel N) → per_invoice: extract → validate → mark_done
                                    → aggregate_results → update_job → dispatch_webhook
```

### Agent 4 — Validation Agent (new)
LangGraph. Runs after every extraction. Checks: arithmetic (subtotal+tax≠total), line item sum, duplicate invoice number, future/stale date, amount anomaly (>3σ from tenant 90-day history), currency mismatch.
```
load_extraction → arithmetic_check → duplicate_check → anomaly_check → write_report
```

### Agent 5 — Discrepancy Detection Agent (upgraded)
Two-phase: deterministic field comparison (existing logic) + Gemini semantic analysis for ambiguous cases (vendor aliases, fuzzy duplicate detection). Returns `DiscrepancyReport` with severity: `info | warning | critical`.
```
load_invoices → structural_diff → semantic_analysis (Gemini) → risk_scoring → report
```

### Agent 6 — Webhook Dispatcher (new)
Simple RQ job (not LangGraph). HMAC-SHA256 signing. Exponential backoff: 5s, 30s, 2m, 10m, 30m. Max 5 attempts. Every attempt written to `webhook_deliveries`. Admin notified on final failure.

### LLM Provider Abstraction
```python
class LLMProvider(Protocol):
    def embed_text(self, texts: list[str]) -> list[list[float]]: ...
    def embed_image(self, image_path: Path) -> list[float]: ...
    def generate(self, prompt: str) -> str: ...
    def generate_structured(self, prompt: str, schema: type) -> dict: ...

# Config-driven: LLM_PROVIDER=gemini (prod) | LLM_PROVIDER=ollama_gemma (dev)
```

### Embedding Strategy
- **Production:** `gemini-embedding-exp-03-07` (768 dims) for text; Gemini 2.0 Flash vision → text → embed for images
- **Local dev:** Gemma 3 4B via Ollama (multimodal, covers both text and image natively)
- **pgvector:** IVFFlat index, cosine distance, 768 dims

---

## 7. Auth, RBAC & Security

### Auth Flow
- **Users:** Supabase Auth JWT (`Authorization: Bearer <jwt>`). JWT carries `sub`, `role`, `tenant_id`.
- **API users:** `X-API-Key` header → bcrypt hash lookup → synthetic `CurrentUser` injected.
- **Supabase RLS:** Every table enforces `tenant_id = auth.jwt() ->> 'tenant_id'` — cross-tenant leaks impossible at DB layer.

### Security Controls
| Control | Implementation |
|---|---|
| Rate limiting | slowapi: 60 req/min per tenant; 10/min on `/upload` |
| File validation | python-magic mime check + 50MB max size |
| CORS | Restricted to `ALLOWED_ORIGINS` env var |
| Secrets | Render environment vars only — no .env in production |
| Webhook signing | HMAC-SHA256 per-webhook secret |
| API key storage | bcrypt hash only, raw key shown once |
| SQL injection | SQLAlchemy ORM + parameterised queries |
| Path traversal | All file I/O via Supabase Storage SDK |
| Audit trail | Append-only `audit_log`, RLS blocks DELETE for all roles |

---

## 8. Observability

### Stack
| Tool | Purpose | Trigger |
|---|---|---|
| structlog | JSON structured logs with request_id/tenant_id/user_id on every line | Every request + job |
| LangSmith | Full LLM traces for all agents: prompts, responses, latency, token counts | Every agent invocation |
| Sentry | Unhandled exceptions + p95 slow endpoint alerts (threshold: 3s) | On exception or breach |
| Prometheus | HTTP metrics + custom: extractions_total, extraction_duration, tokens_used_total | Scraped every 15s |
| llm_usage table | Per-tenant token + cost tracking for billing visibility | After every Gemini call |
| audit_log table | Security + compliance trail for every write action | Every mutating request |

### LangSmith Configuration
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=invoice-analyst-prod
```
Custom metadata per trace: `tenant_id`, `invoice_id`, `env`. Used for evals, prompt versioning via LangSmith Hub, and analyst feedback loop → fine-tuning dataset.

### Custom Prometheus Metrics
```python
extractions_total       Counter  labels: [tenant_id, model, status]
extraction_duration     Histogram  buckets: [0.5, 1, 2, 5, 10, 30]s
tokens_used_total       Counter  labels: [tenant_id, model, direction]
jobs_queued             Counter  labels: [type]
```

---

## 9. Project Structure

```
invoice-extractor/
├── api/              # FastAPI backend
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── routers/      # invoices, extraction, qa, compare, batch, jobs,
│   │                 # webhooks, users, api_keys, audit, health
│   ├── middleware/   # request_context, rate_limiter, audit_writer
│   ├── services/     # storage, embeddings, llm_tracker, webhook_signer
│   └── schemas/      # Pydantic request/response models
├── agents/           # All LangGraph agents + LLMProvider abstraction
│   ├── base.py
│   ├── providers/    # gemini.py, ollama_gemma.py
│   ├── qa_agent.py
│   ├── extraction_agent.py
│   ├── batch_agent.py
│   ├── validation_agent.py
│   ├── discrepancy_agent.py
│   └── retriever.py  # HybridRetriever (pgvector + BM25)
├── workers/          # RQ background jobs
│   ├── worker.py
│   ├── ingest_job.py
│   ├── extract_job.py
│   ├── batch_job.py
│   └── webhook_job.py
├── db/               # SQLAlchemy models + Alembic migrations
├── frontend/         # Streamlit thin client (calls FastAPI)
│   ├── app.py
│   ├── auth.py
│   ├── api_client.py
│   └── pages/        # qa, extract, compare, batch, dashboard
├── models/           # InvoiceSchema (unchanged)
├── eval/             # Existing eval suite
├── tests/
│   ├── unit/
│   ├── integration/  # real DB, mocked Gemini
│   └── e2e/
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── render.yaml
├── alembic.ini
└── pyproject.toml
```

---

## 10. CI/CD & Deployment

### GitHub Actions
- **`ci.yml`** (every PR): ruff lint → mypy type check → pytest (unit + integration, real Postgres + Redis in GH Actions services) → codecov
- **`deploy.yml`** (push to main): trigger Render deploy hooks → run `alembic upgrade head` against production DB

### Render Services
| Service | Type | Dockerfile |
|---|---|---|
| invoice-api | Web (Render Starter) | Dockerfile.api |
| invoice-frontend | Web (Render Starter) | Dockerfile.frontend |
| invoice-worker | Worker (Render Starter) | Dockerfile.api |
| invoice-redis | Redis (Render Starter) | managed |

### Local Dev Quickstart
```bash
cp .env.example .env.local       # fill in GOOGLE_API_KEY, Supabase keys
ollama pull gemma3:4b             # local LLM + embeddings (LLM_PROVIDER=ollama_gemma)
docker compose up                 # api + frontend + worker + redis + postgres
alembic upgrade head              # run migrations
# API docs:  http://localhost:8000/docs
# Frontend:  http://localhost:8501
# LangSmith: https://smith.langchain.com
```

### Environment Variables (production, set in Render)
```
SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY
GOOGLE_API_KEY
LLM_PROVIDER=gemini
REDIS_URL                         (auto-wired from Render Redis service)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY
LANGCHAIN_PROJECT=invoice-analyst-prod
SENTRY_DSN
ENV=production
GIT_SHA                           (auto-set by Render)
ALLOWED_ORIGINS
```

---

## 11. What Is NOT Changing

- `models/invoice.py` — `InvoiceSchema` and `LineItem` are unchanged
- `rag/comparator.py` — deterministic comparison logic reused inside Discrepancy Agent Phase 1
- `eval/` — existing eval suite reused with pgvector retriever
- Core LangGraph graph topology — nodes and edges unchanged, only provider backends swap

---

## 12. Open Questions (resolved at implementation time)

- Supabase pgvector IVFFlat `lists` parameter: start at 100, tune based on chunk volume
- Render Starter plan memory limits: monitor worker RSS during batch jobs; upgrade if >512MB
- LangSmith Hub prompt versioning: adopt after initial deployment, not a blocker
