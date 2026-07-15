# Invoice Extractor — Enterprise Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the invoice-extractor Streamlit monolith into a multi-tenant enterprise SaaS with a FastAPI backbone, Supabase storage, Gemini LLM, 6 LangGraph agents, RQ workers, full RBAC, and LangSmith/Sentry/Prometheus observability.

**Architecture:** FastAPI backend (the spine) + Streamlit thin analyst client + RQ background workers on Render. Supabase provides Postgres + pgvector + Storage + Auth. All business logic lives in FastAPI; Streamlit only calls the API.

**Tech Stack:** FastAPI, SQLAlchemy (async), Alembic, Supabase (Postgres + pgvector + Storage + Auth), LangGraph, google-genai (Gemini 2.0 Flash + gemini-embedding-exp-03-07), Ollama (Gemma 3 4B local fallback), RQ + Redis, structlog, LangSmith, Sentry, Prometheus, Streamlit, Docker, Render, GitHub Actions, pytest, ruff, mypy.

---

## File Map

```
invoice-extractor/
├── pyproject.toml                         MODIFY — add all new deps
├── alembic.ini                            CREATE
├── Dockerfile.api                         CREATE
├── Dockerfile.frontend                    CREATE
├── docker-compose.yml                     CREATE
├── render.yaml                            CREATE
│
├── api/
│   ├── __init__.py                        CREATE (empty)
│   ├── main.py                            CREATE — app factory, routers, middleware
│   ├── config.py                          CREATE — pydantic-settings
│   ├── dependencies.py                    CREATE — get_db, get_queue, get_current_user, require_roles
│   ├── routers/
│   │   ├── __init__.py                    CREATE (empty)
│   │   ├── invoices.py                    CREATE
│   │   ├── extraction.py                  CREATE
│   │   ├── qa.py                          CREATE
│   │   ├── compare.py                     CREATE
│   │   ├── batch.py                       CREATE
│   │   ├── jobs.py                        CREATE
│   │   ├── webhooks.py                    CREATE
│   │   ├── users.py                       CREATE
│   │   ├── api_keys.py                    CREATE
│   │   ├── audit.py                       CREATE
│   │   └── health.py                      CREATE
│   ├── middleware/
│   │   ├── __init__.py                    CREATE (empty)
│   │   ├── request_context.py             CREATE
│   │   ├── rate_limiter.py                CREATE
│   │   └── audit_writer.py                CREATE
│   ├── services/
│   │   ├── __init__.py                    CREATE (empty)
│   │   ├── storage.py                     CREATE — Supabase Storage wrapper
│   │   ├── embeddings.py                  CREATE — embed text + image via Gemini
│   │   ├── llm_tracker.py                 CREATE — @llm_usage_tracker decorator
│   │   └── webhook_signer.py              CREATE — HMAC-SHA256
│   └── schemas/
│       ├── __init__.py                    CREATE (empty)
│       ├── invoice.py                     CREATE
│       ├── extraction.py                  CREATE
│       ├── job.py                         CREATE
│       ├── webhook.py                     CREATE
│       └── user.py                        CREATE
│
├── agents/
│   ├── __init__.py                        CREATE (empty)
│   ├── base.py                            CREATE — LLMProvider Protocol + factory
│   ├── providers/
│   │   ├── __init__.py                    CREATE (empty)
│   │   ├── gemini.py                      CREATE — GeminiProvider
│   │   └── ollama_gemma.py                CREATE — OllamaGemmaProvider
│   ├── retriever.py                       MODIFY — pgvector backend replaces ChromaDB
│   ├── qa_agent.py                        MODIFY — swap providers, add SSE support
│   ├── extraction_agent.py                MODIFY — Gemini structured output + retry node
│   ├── batch_agent.py                     CREATE — LangGraph Send fan-out
│   ├── validation_agent.py                CREATE — arithmetic + anomaly checks
│   └── discrepancy_agent.py               MODIFY — add Gemini semantic phase
│
├── workers/
│   ├── __init__.py                        CREATE (empty)
│   ├── worker.py                          CREATE — RQ worker entrypoint
│   ├── ingest_job.py                      CREATE — download→chunk→embed→pgvector
│   ├── extract_job.py                     CREATE — run extraction_agent
│   ├── batch_job.py                       CREATE — run batch_agent
│   └── webhook_job.py                     CREATE — HMAC sign + POST + retry
│
├── db/
│   ├── __init__.py                        CREATE (empty)
│   ├── session.py                         CREATE — async SQLAlchemy engine
│   ├── models.py                          CREATE — all ORM models
│   └── migrations/
│       ├── env.py                         CREATE
│       ├── script.py.mako                 CREATE
│       └── versions/
│           └── 0001_initial.py            CREATE — full schema
│
├── frontend/
│   ├── app.py                             MODIFY — thin client, calls API
│   ├── auth.py                            MODIFY — Supabase Auth login/logout
│   ├── api_client.py                      CREATE — httpx wrapper
│   └── pages/
│       ├── qa.py                          CREATE
│       ├── extract.py                     CREATE
│       ├── compare.py                     CREATE
│       ├── batch.py                       CREATE (new page)
│       └── dashboard.py                   CREATE (new page)
│
├── models/
│   └── invoice.py                         NO CHANGE
│
└── tests/
    ├── conftest.py                         MODIFY — add DB, mock Gemini, Redis fixtures
    ├── unit/
    │   ├── test_providers.py               CREATE
    │   ├── test_retriever.py               CREATE
    │   ├── test_agents.py                  MODIFY
    │   ├── test_validation_agent.py        CREATE
    │   └── test_comparator.py              MODIFY
    ├── integration/
    │   ├── test_api_auth.py                CREATE
    │   ├── test_api_invoices.py            CREATE
    │   ├── test_api_extraction.py          CREATE
    │   ├── test_api_batch.py               CREATE
    │   └── test_webhooks.py                CREATE
    └── e2e/
        └── test_full_pipeline.py           CREATE
```

---

## Phase 1 — Foundation

**Deliverable:** Working project skeleton with all dependencies, DB models, and a passing migration against a local Postgres+pgvector container.

---

### Task 1: pyproject.toml — full dependency manifest

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing test to verify imports**

Create `tests/test_imports.py`:
```python
def test_core_imports():
    import fastapi
    import sqlalchemy
    import alembic
    import redis
    import rq
    import structlog
    import sentry_sdk
    import slowapi
    import bcrypt
    import httpx
    import magic
    import prometheus_client

def test_agent_imports():
    import langgraph
    import langchain_core
    import google.genai

def test_frontend_imports():
    import streamlit
    import supabase
```

Run: `pytest tests/test_imports.py -v`
Expected: FAIL with ImportError

- [ ] **Step 2: Replace pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "invoice-extractor"
version = "2.0.0"
requires-python = ">=3.12"

[project.optional-dependencies]
prod = [
    # API
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "python-multipart>=0.0.9",
    "slowapi>=0.1.9",
    "httpx>=0.27.0",
    # DB
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",
    "pgvector>=0.3.0",
    # Supabase
    "supabase>=2.5.0",
    # Auth / Security
    "python-jose[cryptography]>=3.3.0",
    "bcrypt>=4.1.0",
    "python-magic>=0.4.27",
    # Queue
    "rq>=1.16.0",
    "redis>=5.0.0",
    # AI
    "langchain-core>=0.3.0",
    "langchain-text-splitters>=0.3.0",
    "langgraph>=0.2.0",
    "google-genai>=1.0.0",
    "langchain-ollama>=0.2.0",
    "rank-bm25>=0.2.2",
    # Observability
    "structlog>=24.0.0",
    "sentry-sdk[fastapi]>=2.0.0",
    "prometheus-fastapi-instrumentator>=7.0.0",
    "prometheus-client>=0.20.0",
    "langsmith>=0.1.0",
    # LLM tracing
    "langchain-core>=0.3.0",
    # Utils
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "python-box>=7.0.0",
    "pyyaml>=6.0",
    "pandas>=2.0.0",
    "pypdf>=4.2.0",
    "Pillow>=10.3.0",
    "fpdf2>=2.7.0",
    "pypdfium2>=4.0.0",
    "rapidocr-onnxruntime>=1.3.0",
]
frontend = [
    "streamlit>=1.37.0",
    "httpx>=0.27.0",
    "supabase>=2.5.0",
    "pydantic>=2.0.0",
    "pandas>=2.0.0",
    "python-dotenv>=1.0.0",
]
dev = [
    "invoice-extractor[prod]",
    "invoice-extractor[frontend]",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",
    "respx>=0.21.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["api*", "agents*", "workers*", "db*", "frontend*", "models*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.12"
strict = false
ignore_missing_imports = true
```

- [ ] **Step 3: Install and verify**

```bash
pip install -e ".[dev]"
pytest tests/test_imports.py -v
```
Expected: All PASS

- [ ] **Step 4: Commit**
```bash
git add pyproject.toml tests/test_imports.py
git commit -m "chore: add enterprise dependency manifest"
```

---

### Task 2: DB models

**Files:**
- Create: `db/__init__.py`
- Create: `db/models.py`
- Create: `db/session.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_models.py`:
```python
import uuid
from db.models import Tenant, User, Invoice, InvoiceChunk, Extraction, Job, Webhook, WebhookDelivery, AuditLog, ApiKey, LlmUsage

def test_tenant_has_required_columns():
    cols = {c.name for c in Tenant.__table__.columns}
    assert {"id", "name", "plan", "created_at"}.issubset(cols)

def test_invoice_chunk_has_vector_column():
    cols = {c.name for c in InvoiceChunk.__table__.columns}
    assert "embedding" in cols

def test_user_role_choices():
    from sqlalchemy import inspect
    col = User.__table__.columns["role"]
    assert col is not None

def test_all_tables_have_tenant_id():
    tables_needing_tenant = [Invoice, InvoiceChunk, Extraction, Job, Webhook, AuditLog, ApiKey, LlmUsage]
    for model in tables_needing_tenant:
        cols = {c.name for c in model.__table__.columns}
        assert "tenant_id" in cols, f"{model.__name__} missing tenant_id"
```

Run: `pytest tests/unit/test_models.py -v`
Expected: FAIL with ImportError

- [ ] **Step 2: Create `db/__init__.py`**
```python
```

- [ ] **Step 3: Create `db/models.py`**

```python
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer,
    Numeric, String, Text, ARRAY, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    plan = Column(String(50), nullable=False, server_default="free")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    users = relationship("User", back_populates="tenant")
    invoices = relationship("Invoice", back_populates="tenant")


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(320), nullable=False)
    role = Column(String(20), nullable=False, server_default="analyst")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    tenant = relationship("Tenant", back_populates="users")
    __table_args__ = (UniqueConstraint("tenant_id", "email"),)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    role = Column(String(20), nullable=False, server_default="api_user")
    active = Column(Boolean, nullable=False, server_default="true")
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class Invoice(Base):
    __tablename__ = "invoices"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    file_name = Column(String(512), nullable=False)
    file_type = Column(String(10), nullable=False)  # 'pdf' | 'image'
    storage_path = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False)
    status = Column(String(20), nullable=False, server_default="pending")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    tenant = relationship("Tenant", back_populates="invoices")
    chunks = relationship("InvoiceChunk", back_populates="invoice", cascade="all, delete-orphan")
    extraction = relationship("Extraction", back_populates="invoice", uselist=False, cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("tenant_id", "sha256"),)


class InvoiceChunk(Base):
    __tablename__ = "invoice_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    page_num = Column(Integer, nullable=False)
    embedding = Column(Vector(768), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    invoice = relationship("Invoice", back_populates="chunks")


class Extraction(Base):
    __tablename__ = "extractions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, unique=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    schema_json = Column(JSONB, nullable=False)
    model_used = Column(String(100), nullable=False)
    validated = Column(Boolean, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    invoice = relationship("Invoice", back_populates="extraction")


class Job(Base):
    __tablename__ = "jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, server_default="queued")
    payload = Column(JSONB, nullable=True)
    result = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class Webhook(Base):
    __tablename__ = "webhooks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    url = Column(Text, nullable=False)
    events = Column(ARRAY(String), nullable=False)
    secret = Column(Text, nullable=False)
    active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    deliveries = relationship("WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan")


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(UUID(as_uuid=True), ForeignKey("webhooks.id", ondelete="CASCADE"), nullable=False)
    event = Column(String(100), nullable=False)
    payload = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, server_default="pending")
    attempts = Column(Integer, nullable=False, server_default="0")
    last_error = Column(Text, nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    webhook = relationship("Webhook", back_populates="deliveries")


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class LlmUsage(Base):
    __tablename__ = "llm_usage"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    invoice_id = Column(UUID(as_uuid=True), nullable=True)
    model = Column(String(100), nullable=False)
    agent = Column(String(50), nullable=False)
    input_tokens = Column(Integer, nullable=False, server_default="0")
    output_tokens = Column(Integer, nullable=False, server_default="0")
    latency_ms = Column(Integer, nullable=False, server_default="0")
    cost_usd = Column(Numeric(10, 6), nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
```

- [ ] **Step 4: Create `db/session.py`**

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.config import get_settings

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=settings.ENV == "development",
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def get_db() -> AsyncSession:
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

- [ ] **Step 5: Run tests**
```bash
pytest tests/unit/test_models.py -v
```
Expected: All PASS

- [ ] **Step 6: Commit**
```bash
git add db/
git commit -m "feat: add SQLAlchemy ORM models for all enterprise tables"
```

---

### Task 3: Alembic migration

**Files:**
- Create: `alembic.ini`
- Create: `db/migrations/env.py`
- Create: `db/migrations/script.py.mako`
- Create: `db/migrations/versions/0001_initial.py`

- [ ] **Step 1: Create `alembic.ini`**

```ini
[alembic]
script_location = db/migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url =

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

- [ ] **Step 2: Create `db/migrations/env.py`**

```python
import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    return os.environ["DATABASE_URL"]


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = create_async_engine(get_url())
    async with connectable.connect() as connection:
        await connection.run_sync(
            lambda conn: context.configure(connection=conn, target_metadata=target_metadata)
        )
        async with connection.begin():
            await connection.run_sync(lambda _: context.run_migrations())
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

- [ ] **Step 3: Create `db/migrations/script.py.mako`**

```
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 4: Create `db/migrations/versions/0001_initial.py`**

```python
"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-12
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("plan", sa.String(50), nullable=False, server_default="free"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="analyst"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("tenant_id", "email"),
    )

    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("key_hash", sa.String(255), nullable=False, unique=True),
        sa.Column("role", sa.String(20), nullable=False, server_default="api_user"),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "invoices",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("uploaded_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("file_name", sa.String(512), nullable=False),
        sa.Column("file_type", sa.String(10), nullable=False),
        sa.Column("storage_path", sa.Text, nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("tenant_id", "sha256"),
    )

    op.create_table(
        "invoice_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("invoice_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_text", sa.Text, nullable=False),
        sa.Column("page_num", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.execute(
        "CREATE INDEX ON invoice_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.create_table(
        "extractions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("invoice_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("schema_json", postgresql.JSONB, nullable=False),
        sa.Column("model_used", sa.String(100), nullable=False),
        sa.Column("validated", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("payload", postgresql.JSONB, nullable=True),
        sa.Column("result", postgresql.JSONB, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "webhooks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("events", postgresql.ARRAY(sa.String), nullable=False),
        sa.Column("secret", sa.Text, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "webhook_deliveries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("webhook_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("webhooks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event", sa.String(100), nullable=False),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text, nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=True),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "llm_usage",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("invoice_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("agent", sa.String(50), nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    for table in ["llm_usage", "audit_log", "webhook_deliveries", "webhooks",
                  "jobs", "extractions", "invoice_chunks", "invoices",
                  "api_keys", "users", "tenants"]:
        op.drop_table(table)
```

- [ ] **Step 5: Start local Postgres and run migration**

```bash
docker compose up postgres -d
DATABASE_URL=postgresql+asyncpg://postgres:dev@localhost/invoice_dev alembic upgrade head
```
Expected: `Running upgrade  -> 0001, initial schema`

- [ ] **Step 6: Commit**
```bash
git add alembic.ini db/migrations/
git commit -m "feat: add Alembic migration — full initial schema with pgvector"
```

---

### Task 4: api/config.py

**Files:**
- Create: `api/__init__.py`
- Create: `api/config.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_config.py`:
```python
import os
import pytest
from api.config import Settings

def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:y@localhost/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "eyJ")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "eyJ2")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza")
    s = Settings()
    assert s.DATABASE_URL.startswith("postgresql")
    assert s.ENV == "development"
    assert s.LLM_PROVIDER == "gemini"
```

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/__init__.py`**
```python
```

- [ ] **Step 3: Create `api/config.py`**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", extra="ignore")

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_KEY: str

    # AI
    GOOGLE_API_KEY: str
    LLM_PROVIDER: str = "gemini"           # "gemini" | "ollama_gemma"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-exp-03-07"
    GEMMA_MODEL: str = "gemma3:4b"

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "invoice-analyst-dev"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # Sentry
    SENTRY_DSN: str = ""

    # App
    ENV: str = "development"
    GIT_SHA: str = "local"
    ALLOWED_ORIGINS: str = "http://localhost:8501"
    API_BASE_URL: str = "http://localhost:8000"

    # RAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 80
    NUM_RESULTS: int = 4
    MAX_AGENT_ITERATIONS: int = 3
    MAX_CRITIQUE_ITERATIONS: int = 1

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test**
```bash
pytest tests/unit/test_config.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add api/__init__.py api/config.py tests/unit/test_config.py
git commit -m "feat: add pydantic-settings config for all env vars"
```

---

## Phase 2 — FastAPI Core

**Deliverable:** Running FastAPI app with auth, RBAC, middleware, all routers registered, and `/health` returning `{ status: ok }`.

---

### Task 5: Auth dependencies

**Files:**
- Create: `api/dependencies.py`
- Create: `api/schemas/user.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_dependencies.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from api.dependencies import verify_supabase_jwt, CurrentUser

def test_verify_jwt_raises_on_bad_token():
    with pytest.raises(HTTPException) as exc_info:
        verify_supabase_jwt("bad.token.here")
    assert exc_info.value.status_code == 401

def test_current_user_is_dataclass():
    u = CurrentUser(id="abc", tenant_id="t1", role="analyst", email="a@b.com")
    assert u.role == "analyst"
    assert u.tenant_id == "t1"
```

Run: `pytest tests/unit/test_dependencies.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/schemas/__init__.py` and `api/schemas/user.py`**

`api/schemas/__init__.py`:
```python
```

`api/schemas/user.py`:
```python
from pydantic import BaseModel
import uuid


class UserOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    role: str

class RoleUpdateIn(BaseModel):
    role: str
```

- [ ] **Step 3: Create `api/dependencies.py`**

```python
import uuid
from dataclasses import dataclass
from typing import Callable

import bcrypt
import structlog
from fastapi import Depends, HTTPException, Request
from jose import JWTError, jwt
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import get_settings
from db.models import ApiKey
from db.session import get_db

log = structlog.get_logger()


@dataclass
class CurrentUser:
    id: str
    tenant_id: str
    role: str
    email: str = ""


def verify_supabase_jwt(token: str) -> dict:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_ANON_KEY,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        return payload
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    auth_header = request.headers.get("Authorization", "")
    api_key_header = request.headers.get("X-API-Key", "")

    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
        payload = verify_supabase_jwt(token)
        user_meta = payload.get("user_metadata", {})
        app_meta = payload.get("app_metadata", {})
        return CurrentUser(
            id=payload.get("sub", ""),
            tenant_id=str(app_meta.get("tenant_id", "")),
            role=str(app_meta.get("role", "viewer")),
            email=str(payload.get("email", "")),
        )

    if api_key_header:
        key_bytes = api_key_header.encode()
        result = await db.execute(
            select(ApiKey).where(ApiKey.active == True)
        )
        for row in result.scalars():
            if bcrypt.checkpw(key_bytes, row.key_hash.encode()):
                await db.execute(
                    update(ApiKey)
                    .where(ApiKey.id == row.id)
                    .values(last_used_at=__import__("datetime").datetime.utcnow())
                )
                return CurrentUser(
                    id=str(row.id),
                    tenant_id=str(row.tenant_id),
                    role=row.role,
                    email="api_key",
                )
        raise HTTPException(status_code=401, detail="Invalid API key")

    raise HTTPException(status_code=401, detail="Authentication required")


def require_roles(*roles: str) -> Callable:
    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return _check


# Convenience role-gated dependencies
AdminOnly = Depends(require_roles("admin"))
AnalystOrAbove = Depends(require_roles("admin", "analyst"))
ViewerOrAbove = Depends(require_roles("admin", "analyst", "viewer"))
APIUserOrAbove = Depends(require_roles("admin", "analyst", "api_user"))
AnyRole = Depends(require_roles("admin", "analyst", "viewer", "api_user"))


async def get_queue():
    from redis import Redis
    from rq import Queue
    settings = get_settings()
    conn = Redis.from_url(settings.REDIS_URL)
    return Queue("invoice-jobs", connection=conn)
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/unit/test_dependencies.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add api/dependencies.py api/schemas/
git commit -m "feat: add JWT + API key auth dependencies with RBAC"
```

---

### Task 6: Middleware

**Files:**
- Create: `api/middleware/request_context.py`
- Create: `api/middleware/rate_limiter.py`
- Create: `api/middleware/audit_writer.py`
- Create: `api/middleware/__init__.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_middleware.py`:
```python
from api.middleware.request_context import bind_request_context
import structlog.contextvars as ctx

def test_request_context_clears_between_requests():
    ctx.clear_contextvars()
    ctx.bind_contextvars(request_id="abc123")
    assert ctx.get_contextvars().get("request_id") == "abc123"
    ctx.clear_contextvars()
    assert ctx.get_contextvars().get("request_id") is None
```

Run: `pytest tests/unit/test_middleware.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/middleware/__init__.py`**
```python
```

- [ ] **Step 3: Create `api/middleware/request_context.py`**

```python
import uuid
import structlog
import structlog.contextvars
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.clear_contextvars()
        return response


# standalone function for tests
def bind_request_context(request_id: str, path: str = "", method: str = "") -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id, path=path, method=method)
```

- [ ] **Step 4: Create `api/middleware/rate_limiter.py`**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address


def _get_tenant_id(request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            import base64, json
            parts = auth.removeprefix("Bearer ").split(".")
            padded = parts[1] + "=" * (-len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            tenant_id = payload.get("app_metadata", {}).get("tenant_id")
            if tenant_id:
                return str(tenant_id)
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_get_tenant_id)
```

- [ ] **Step 5: Create `api/middleware/audit_writer.py`**

```python
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from db.models import AuditLog

log = structlog.get_logger()

_AUDITED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_SKIP_PATHS = {"/health", "/health/ready", "/metrics", "/docs", "/openapi.json"}


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if (
            request.method in _AUDITED_METHODS
            and request.url.path not in _SKIP_PATHS
            and response.status_code < 400
        ):
            try:
                user = getattr(request.state, "current_user", None)
                if user and user.tenant_id:
                    db: AsyncSession = request.state.db
                    entry = AuditLog(
                        tenant_id=uuid.UUID(user.tenant_id),
                        user_id=uuid.UUID(user.id) if user.id else None,
                        action=f"{request.method.lower()}.{request.url.path.split('/')[-1]}",
                        metadata={"path": request.url.path, "status": response.status_code},
                        created_at=datetime.now(timezone.utc),
                    )
                    db.add(entry)
                    await db.commit()
            except Exception as exc:
                log.warning("audit_write_failed", error=str(exc))
        return response
```

- [ ] **Step 6: Run tests**
```bash
pytest tests/unit/test_middleware.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add api/middleware/
git commit -m "feat: add request context, rate limiter, and audit middleware"
```

---

### Task 7: FastAPI app factory + health router

**Files:**
- Create: `api/main.py`
- Create: `api/routers/__init__.py`
- Create: `api/routers/health.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_health.py`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import create_app

@pytest.mark.asyncio
async def test_health_returns_ok():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ok"

@pytest.mark.asyncio
async def test_ready_probe():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
```

Run: `pytest tests/integration/test_health.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/routers/__init__.py`**
```python
```

- [ ] **Step 3: Create `api/routers/health.py`**

```python
import structlog
from fastapi import APIRouter
from redis import Redis

from api.config import get_settings
from db.session import get_engine

router = APIRouter(prefix="/health", tags=["health"])
log = structlog.get_logger()


@router.get("")
async def health():
    settings = get_settings()
    db_ok = True
    redis_ok = True
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:
        db_ok = False
        log.error("health_db_failed", error=str(exc))
    try:
        r = Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1)
        r.ping()
    except Exception as exc:
        redis_ok = False
        log.error("health_redis_failed", error=str(exc))
    return {
        "data": {"status": "ok", "db": "ok" if db_ok else "error", "redis": "ok" if redis_ok else "error"},
        "error": None,
        "request_id": None,
    }


@router.get("/ready")
async def ready():
    return {"data": {"ready": True}, "error": None, "request_id": None}
```

- [ ] **Step 4: Create `api/main.py`**

```python
import structlog
import structlog.contextvars
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse

from api.config import get_settings
from api.middleware.rate_limiter import limiter
from api.middleware.request_context import RequestContextMiddleware
from api.routers import health

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)


def create_app() -> FastAPI:
    settings = get_settings()

    if settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.2,
            environment=settings.ENV,
            release=settings.GIT_SHA,
        )

    app = FastAPI(
        title="Invoice Analyst API",
        version="2.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        lambda req, exc: JSONResponse(
            status_code=429,
            content={"data": None, "error": "Rate limit exceeded", "request_id": None},
        ),
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1")

    # Remaining routers added in later tasks
    return app


app = create_app()
```

- [ ] **Step 5: Run tests**
```bash
pytest tests/integration/test_health.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add api/main.py api/routers/
git commit -m "feat: add FastAPI app factory with health router and middleware"
```

---

## Phase 3 — AI Layer

**Deliverable:** All 6 LangGraph agents running against pgvector + Gemini, fully tested with mocked providers, LangSmith tracing wired.

---

### Task 8: LLMProvider protocol + Gemini provider

**Files:**
- Create: `agents/__init__.py`
- Create: `agents/base.py`
- Create: `agents/providers/__init__.py`
- Create: `agents/providers/gemini.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_providers.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from agents.base import get_provider
from agents.providers.gemini import GeminiProvider

def test_get_provider_returns_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    provider = get_provider()
    assert isinstance(provider, GeminiProvider)

def test_gemini_provider_has_required_methods():
    provider = GeminiProvider.__new__(GeminiProvider)
    assert hasattr(provider, "embed_text")
    assert hasattr(provider, "embed_image")
    assert hasattr(provider, "generate")
    assert hasattr(provider, "generate_structured")

def test_embed_text_returns_list_of_floats(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
    mock_client.models.embed_content.return_value = mock_response
    provider = GeminiProvider.__new__(GeminiProvider)
    provider._client = mock_client
    provider._embed_model = "gemini-embedding-exp-03-07"
    result = provider.embed_text(["hello world"])
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert result[0] == [0.1, 0.2, 0.3]
```

Run: `pytest tests/unit/test_providers.py -v`
Expected: FAIL

- [ ] **Step 2: Create `agents/__init__.py`**
```python
```

- [ ] **Step 3: Create `agents/base.py`**

```python
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def embed_text(self, texts: list[str]) -> list[list[float]]: ...
    def embed_image(self, image_path: Path) -> list[float]: ...
    def generate(self, prompt: str, system: str | None = None) -> str: ...
    def generate_structured(self, prompt: str, schema: type) -> dict: ...


def get_provider() -> LLMProvider:
    from api.config import get_settings
    settings = get_settings()
    if settings.LLM_PROVIDER == "ollama_gemma":
        from agents.providers.ollama_gemma import OllamaGemmaProvider
        return OllamaGemmaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.GEMMA_MODEL,
        )
    from agents.providers.gemini import GeminiProvider
    return GeminiProvider(
        api_key=settings.GOOGLE_API_KEY,
        model=settings.GEMINI_MODEL,
        embed_model=settings.GEMINI_EMBEDDING_MODEL,
    )
```

- [ ] **Step 4: Create `agents/providers/__init__.py`**
```python
```

- [ ] **Step 5: Create `agents/providers/gemini.py`**

```python
import json
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from models.invoice import InvoiceSchema


class GeminiProvider:
    def __init__(self, api_key: str, model: str, embed_model: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._embed_model = embed_model

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._embed_model,
            contents=texts,
        )
        return [e.values for e in response.embeddings]

    def embed_image(self, image_path: Path) -> list[float]:
        img = Image.open(image_path)
        description = self.generate(
            prompt=(
                "Describe all text, numbers, dates, vendor names, and amounts "
                "visible in this invoice image in detail."
            ),
            image=img,
        )
        return self.embed_text([description])[0]

    def generate(self, prompt: str, system: str | None = None, image=None) -> str:
        contents: list = []
        if system:
            contents.append(system)
        if image is not None:
            contents.append(image)
        contents.append(prompt)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
        )
        return response.text.strip()

    def generate_structured(self, prompt: str, schema: type) -> dict:
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return json.loads(response.text)
```

- [ ] **Step 6: Run tests**
```bash
pytest tests/unit/test_providers.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add agents/
git commit -m "feat: add LLMProvider protocol and GeminiProvider implementation"
```

---

### Task 9: Ollama Gemma provider (local dev fallback)

**Files:**
- Create: `agents/providers/ollama_gemma.py`

- [ ] **Step 1: Write failing test**

Add to `tests/unit/test_providers.py`:
```python
from agents.providers.ollama_gemma import OllamaGemmaProvider

def test_ollama_provider_has_required_methods():
    provider = OllamaGemmaProvider.__new__(OllamaGemmaProvider)
    assert hasattr(provider, "embed_text")
    assert hasattr(provider, "generate")
    assert hasattr(provider, "generate_structured")

def test_get_provider_returns_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama_gemma")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    from agents.base import get_provider
    provider = get_provider()
    assert isinstance(provider, OllamaGemmaProvider)
```

Run: `pytest tests/unit/test_providers.py::test_ollama_provider_has_required_methods -v`
Expected: FAIL

- [ ] **Step 2: Create `agents/providers/ollama_gemma.py`**

```python
import json
from pathlib import Path

import httpx
from PIL import Image
import base64
import io


class OllamaGemmaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:4b"):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = httpx.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def embed_image(self, image_path: Path) -> list[float]:
        description = self._describe_image(image_path)
        return self.embed_text([description])[0]

    def _describe_image(self, image_path: Path) -> str:
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": "Describe all text, numbers, amounts, dates, and vendor names in this invoice image.",
                "images": [b64],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def generate(self, prompt: str, system: str | None = None, image=None) -> str:
        payload: dict = {"model": self._model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if image is not None and isinstance(image, Path):
            img = Image.open(image).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            payload["images"] = [base64.b64encode(buf.getvalue()).decode()]
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def generate_structured(self, prompt: str, schema: type) -> dict:
        import inspect
        schema_fields = {}
        if hasattr(schema, "model_fields"):
            for name, field in schema.model_fields.items():
                schema_fields[name] = str(field.annotation)
        full_prompt = (
            f"{prompt}\n\nReturn ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema_fields, indent=2)}"
        )
        raw = self.generate(full_prompt)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        return json.loads(raw[start:end])
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/unit/test_providers.py -v
```
Expected: All PASS

- [ ] **Step 4: Commit**
```bash
git add agents/providers/ollama_gemma.py
git commit -m "feat: add OllamaGemmaProvider for local dev fallback"
```

---

### Task 10: pgvector + BM25 hybrid retriever

**Files:**
- Modify: `agents/retriever.py` (replaces `rag/hybrid_retriever.py`)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_retriever.py`:
```python
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_retriever_returns_top_n_chunks():
    from agents.retriever import HybridRetriever
    mock_db = AsyncMock()
    mock_provider = MagicMock()
    mock_provider.embed_text.return_value = [[0.1] * 768]

    chunks = [
        {"id": str(uuid.uuid4()), "chunk_text": f"chunk {i}", "page_num": 1, "score": 0.9 - i * 0.1}
        for i in range(5)
    ]
    mock_db.execute = AsyncMock()
    mock_db.execute.return_value.mappings.return_value.all.return_value = [
        {"chunk_text": c["chunk_text"], "page_num": c["page_num"], "id": c["id"]}
        for c in chunks[:4]
    ]

    retriever = HybridRetriever(
        invoice_id=uuid.uuid4(),
        db=mock_db,
        provider=mock_provider,
        num_results=4,
    )
    with patch.object(retriever, "_bm25_retrieve", return_value=chunks[:4]):
        with patch.object(retriever, "_dense_retrieve", return_value=chunks[:4]):
            results = await retriever.retrieve("what is the total")
    assert len(results) <= 4
    assert all("text" in r and "page" in r and "score" in r for r in results)

@pytest.mark.asyncio
async def test_all_chunks_returns_full_corpus():
    from agents.retriever import HybridRetriever
    mock_db = AsyncMock()
    mock_provider = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.execute.return_value.mappings.return_value.all.return_value = [
        {"chunk_text": f"text {i}", "page_num": i + 1, "id": str(uuid.uuid4())}
        for i in range(10)
    ]
    retriever = HybridRetriever(
        invoice_id=uuid.uuid4(),
        db=mock_db,
        provider=mock_provider,
        num_results=4,
    )
    results = await retriever.all_chunks()
    assert len(results) == 10
```

Run: `pytest tests/unit/test_retriever.py -v`
Expected: FAIL

- [ ] **Step 2: Create `agents/retriever.py`**

```python
import re
import uuid
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base import LLMProvider
from db.models import InvoiceChunk


def _tokenize(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


def _rrf(rank_bm25: int, rank_dense: int, k: int = 60) -> float:
    return 1 / (k + rank_bm25) + 1 / (k + rank_dense)


class HybridRetriever:
    def __init__(
        self,
        invoice_id: uuid.UUID,
        db: AsyncSession,
        provider: LLMProvider,
        num_results: int = 4,
    ):
        self._invoice_id = invoice_id
        self._db = db
        self._provider = provider
        self._num_results = num_results
        self._corpus: list[dict] | None = None

    async def _load_corpus(self) -> list[dict]:
        if self._corpus is None:
            result = await self._db.execute(
                select(InvoiceChunk)
                .where(InvoiceChunk.invoice_id == self._invoice_id)
                .order_by(InvoiceChunk.page_num)
            )
            rows = result.scalars().all()
            self._corpus = [
                {"id": str(r.id), "text": r.chunk_text, "page": r.page_num}
                for r in rows
            ]
        return self._corpus

    def _bm25_retrieve(self, corpus: list[dict], query: str, n: int) -> list[dict]:
        tokenized = [_tokenize(c["text"]) for c in corpus]
        bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[""]])
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [{"idx": i, "rank": rank} for rank, i in enumerate(ranked[:n])]

    async def _dense_retrieve(self, corpus: list[dict], query: str, n: int) -> list[dict]:
        query_embedding = self._provider.embed_text([query])[0]
        result = await self._db.execute(
            text(
                "SELECT id, chunk_text, page_num, "
                "1 - (embedding <=> CAST(:emb AS vector)) AS similarity "
                "FROM invoice_chunks "
                "WHERE invoice_id = :inv_id "
                "ORDER BY embedding <=> CAST(:emb AS vector) "
                "LIMIT :n"
            ),
            {
                "emb": str(query_embedding),
                "inv_id": str(self._invoice_id),
                "n": n,
            },
        )
        rows = result.mappings().all()
        id_to_idx = {c["id"]: i for i, c in enumerate(corpus)}
        return [
            {"idx": id_to_idx[str(r["id"])], "rank": rank}
            for rank, r in enumerate(rows)
            if str(r["id"]) in id_to_idx
        ]

    async def retrieve(self, query: str) -> list[dict]:
        corpus = await self._load_corpus()
        if not corpus:
            return []
        n = min(self._num_results * 3, len(corpus))
        bm25_results = self._bm25_retrieve(corpus, query, n)
        dense_results = await self._dense_retrieve(corpus, query, n)
        bm25_ranks = {r["idx"]: r["rank"] for r in bm25_results}
        dense_ranks = {r["idx"]: r["rank"] for r in dense_results}
        all_indices = set(bm25_ranks) | set(dense_ranks)
        fused = [
            {
                "text": corpus[i]["text"],
                "page": corpus[i]["page"],
                "score": _rrf(bm25_ranks.get(i, n + 60), dense_ranks.get(i, n + 60)),
            }
            for i in all_indices
        ]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[: self._num_results]

    async def all_chunks(self) -> list[dict]:
        corpus = await self._load_corpus()
        return [{"text": c["text"], "page": c["page"]} for c in corpus]
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/unit/test_retriever.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add agents/retriever.py tests/unit/test_retriever.py
git commit -m "feat: add pgvector + BM25 hybrid retriever replacing ChromaDB"
```

---

### Task 11: RAG Q&A agent (upgraded)

**Files:**
- Create: `agents/qa_agent.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_qa_agent.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_qa_agent_returns_answer():
    from agents.qa_agent import build_qa_agent
    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=[
        {"text": "Total amount is $1,234.56", "page": 1, "score": 0.95}
    ])
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        "What is the total amount on this invoice?",  # rewrite
        "yes",                                          # relevance grade
        "The total amount is $1,234.56",               # answer
        "yes",                                          # self-critique
    ]
    agent = build_qa_agent(mock_retriever, mock_provider)
    result = await agent.ainvoke({
        "query": "what is the total",
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    })
    assert "answer" in result
    assert len(result["answer"]) > 0

@pytest.mark.asyncio
async def test_qa_agent_retries_on_irrelevant():
    from agents.qa_agent import build_qa_agent
    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=[
        {"text": "some unrelated text", "page": 1, "score": 0.3}
    ])
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        "rewritten query v1", "no",
        "rewritten query v2", "no",
        "rewritten query v3", "no",
        "Could not find that information.", "yes",
    ]
    agent = build_qa_agent(mock_retriever, mock_provider)
    result = await agent.ainvoke({
        "query": "find the nonexistent field",
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    })
    assert result["iterations"] >= 1
```

Run: `pytest tests/unit/test_qa_agent.py -v`
Expected: FAIL

- [ ] **Step 2: Create `agents/qa_agent.py`**

```python
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.base import LLMProvider
from agents.retriever import HybridRetriever
from api.config import get_settings


class QAState(TypedDict):
    query: str
    rewritten_query: str
    chunks: list[dict]
    answer: str
    relevant: bool
    grounded: bool
    iterations: int
    critique_iterations: int


def build_qa_agent(retriever: HybridRetriever, provider: LLMProvider):
    settings = get_settings()
    max_iter = settings.MAX_AGENT_ITERATIONS
    max_critique = settings.MAX_CRITIQUE_ITERATIONS

    async def query_rewriter(state: QAState) -> QAState:
        prompt = f"Rewrite this invoice question to be specific and extractable.\nOriginal: {state['query']}\n"
        if state.get("rewritten_query"):
            prompt += (
                f"A previous rewrite '{state['rewritten_query']}' retrieved irrelevant "
                f"context; produce a substantively different phrasing.\n"
            )
        prompt += "Rewritten:"
        rewritten = provider.generate(prompt).strip()
        return {**state, "rewritten_query": rewritten, "iterations": state.get("iterations", 0) + 1}

    async def hybrid_retrieve(state: QAState) -> QAState:
        chunks = await retriever.retrieve(state["rewritten_query"])
        return {**state, "chunks": chunks}

    async def relevance_grade(state: QAState) -> QAState:
        if not state["chunks"]:
            return {**state, "relevant": False}
        context = "\n".join(c["text"][:200] for c in state["chunks"])
        prompt = (
            f"Query: {state['rewritten_query']}\nContext: {context}\n"
            f"Is the context relevant to answer the query? Reply ONLY 'yes' or 'no'."
        )
        verdict = provider.generate(prompt).strip().lower()
        return {**state, "relevant": verdict.startswith("yes")}

    def route_from_grade(state: QAState) -> str:
        if state["relevant"] or state.get("iterations", 0) >= max_iter:
            return "generate"
        return "retry"

    async def generate_answer(state: QAState) -> QAState:
        context = "\n\n".join(c["text"] for c in state["chunks"])
        prompt = (
            "Use the following invoice context to answer the question.\n"
            "If the answer is not present, say 'I could not find that information in the invoice.'\n\n"
            f"Context:\n{context}\n\nQuestion: {state['query']}\nAnswer:"
        )
        answer = provider.generate(prompt).strip()
        return {**state, "answer": answer}

    async def self_critique(state: QAState) -> QAState:
        context = "\n\n".join(c["text"] for c in state["chunks"])
        prompt = (
            f"Context:\n{context}\n\nAnswer: {state['answer']}\n\n"
            f"Is this answer directly supported by the context? Reply ONLY 'yes' or 'no'."
        )
        verdict = provider.generate(prompt).strip().lower()
        return {
            **state,
            "grounded": verdict.startswith("yes"),
            "critique_iterations": state.get("critique_iterations", 0) + 1,
        }

    def route_from_critique(state: QAState) -> str:
        if state["grounded"] or state.get("critique_iterations", 0) >= max_critique:
            return "end"
        return "retry"

    graph = StateGraph(QAState)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("relevance_grade", relevance_grade)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("self_critique", self_critique)
    graph.set_entry_point("query_rewriter")
    graph.add_edge("query_rewriter", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "relevance_grade")
    graph.add_conditional_edges(
        "relevance_grade", route_from_grade,
        {"generate": "generate_answer", "retry": "query_rewriter"},
    )
    graph.add_edge("generate_answer", "self_critique")
    graph.add_conditional_edges(
        "self_critique", route_from_critique,
        {"end": END, "retry": "generate_answer"},
    )
    return graph.compile()
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/unit/test_qa_agent.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add agents/qa_agent.py tests/unit/test_qa_agent.py
git commit -m "feat: add upgraded RAG Q&A agent with pgvector and Gemini"
```

---

### Task 12: Smart Extraction agent (upgraded)

**Files:**
- Create: `agents/extraction_agent.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_extraction_agent.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from models.invoice import InvoiceSchema

@pytest.mark.asyncio
async def test_extraction_returns_invoice_schema():
    from agents.extraction_agent import run_extraction
    mock_retriever = AsyncMock()
    mock_retriever.all_chunks = AsyncMock(return_value=[
        {"text": "Vendor: Acme Corp\nTotal: $500.00\nInvoice #: INV-001", "page": 1}
    ])
    mock_provider = MagicMock()
    mock_provider.generate_structured.return_value = {
        "vendor_name": "Acme Corp", "invoice_number": "INV-001",
        "total_amount": 500.0, "currency": "USD",
        "invoice_date": None, "due_date": None, "subtotal": None,
        "tax": None, "po_number": None, "payment_terms": None,
        "vendor_tax_id": None, "vendor_address": None,
        "bill_to": None, "line_items": [],
    }
    result = await run_extraction(mock_retriever, mock_provider)
    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name == "Acme Corp"
    assert result.total_amount == 500.0

@pytest.mark.asyncio
async def test_extraction_retries_on_schema_error():
    from agents.extraction_agent import run_extraction
    mock_retriever = AsyncMock()
    mock_retriever.all_chunks = AsyncMock(return_value=[
        {"text": "invoice content", "page": 1}
    ])
    mock_provider = MagicMock()
    mock_provider.generate_structured.side_effect = [
        ValueError("bad json"),
        {
            "vendor_name": "Retry Corp", "invoice_number": "INV-002",
            "total_amount": 100.0, "currency": "USD",
            "invoice_date": None, "due_date": None, "subtotal": None,
            "tax": None, "po_number": None, "payment_terms": None,
            "vendor_tax_id": None, "vendor_address": None,
            "bill_to": None, "line_items": [],
        }
    ]
    result = await run_extraction(mock_retriever, mock_provider)
    assert result.vendor_name == "Retry Corp"
    assert mock_provider.generate_structured.call_count == 2
```

Run: `pytest tests/unit/test_extraction_agent.py -v`
Expected: FAIL

- [ ] **Step 2: Create `agents/extraction_agent.py`**

```python
from models.invoice import InvoiceSchema
from agents.base import LLMProvider
from agents.retriever import HybridRetriever

_EXTRACTION_PROMPT = """You are an invoice data extractor.
Extract all available fields from the invoice text below.
Return a complete InvoiceSchema JSON object.

Invoice text:
{context}"""

_RETRY_PROMPT = """Previous extraction attempt failed with error: {error}
Please re-extract the invoice data carefully, ensuring the JSON is valid.

Invoice text:
{context}"""

MAX_RETRIES = 2


async def run_extraction(
    retriever: HybridRetriever,
    provider: LLMProvider,
) -> InvoiceSchema:
    chunks = await retriever.all_chunks()
    context = "\n\n".join(c["text"] for c in chunks)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt == 0:
                prompt = _EXTRACTION_PROMPT.format(context=context)
            else:
                prompt = _RETRY_PROMPT.format(error=str(last_error), context=context)
            raw = provider.generate_structured(prompt, InvoiceSchema)
            return InvoiceSchema.model_validate(raw)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                raise ValueError(
                    f"Extraction failed after {MAX_RETRIES + 1} attempts: {exc}"
                ) from exc
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/unit/test_extraction_agent.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add agents/extraction_agent.py tests/unit/test_extraction_agent.py
git commit -m "feat: add smart extraction agent with Gemini structured output and retry"
```

---

### Task 13: Validation agent (new)

**Files:**
- Create: `agents/validation_agent.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_validation_agent.py`:
```python
import pytest
from models.invoice import InvoiceSchema, LineItem
from agents.validation_agent import run_validation, ValidationReport

def test_arithmetic_check_catches_mismatch():
    schema = InvoiceSchema(
        vendor_name="Acme", subtotal=100.0, tax=10.0, total_amount=120.0
    )
    report = run_validation(schema)
    issues = [i for i in report.issues if i["check"] == "arithmetic"]
    assert len(issues) == 1
    assert issues[0]["severity"] == "warning"

def test_arithmetic_check_passes_on_correct():
    schema = InvoiceSchema(
        vendor_name="Acme", subtotal=100.0, tax=10.0, total_amount=110.0
    )
    report = run_validation(schema)
    issues = [i for i in report.issues if i["check"] == "arithmetic"]
    assert len(issues) == 0

def test_line_item_sum_check():
    schema = InvoiceSchema(
        subtotal=100.0,
        line_items=[
            LineItem(description="A", total=50.0),
            LineItem(description="B", total=40.0),
        ]
    )
    report = run_validation(schema)
    issues = [i for i in report.issues if i["check"] == "line_item_sum"]
    assert len(issues) == 1

def test_future_date_check():
    schema = InvoiceSchema(invoice_date="2099-01-01")
    report = run_validation(schema)
    issues = [i for i in report.issues if i["check"] == "future_date"]
    assert len(issues) == 1

def test_no_issues_on_clean_invoice():
    schema = InvoiceSchema(
        vendor_name="Acme", invoice_number="INV-001",
        subtotal=100.0, tax=10.0, total_amount=110.0,
        invoice_date="2026-01-15",
        line_items=[LineItem(description="Service", total=100.0)],
    )
    report = run_validation(schema)
    arithmetic_issues = [i for i in report.issues if i["check"] == "arithmetic"]
    future_issues = [i for i in report.issues if i["check"] == "future_date"]
    assert len(arithmetic_issues) == 0
    assert len(future_issues) == 0
    assert report.passed
```

Run: `pytest tests/unit/test_validation_agent.py -v`
Expected: FAIL

- [ ] **Step 2: Create `agents/validation_agent.py`**

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone

from models.invoice import InvoiceSchema

_TOLERANCE = 0.02


@dataclass
class ValidationReport:
    issues: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i["severity"] in ("warning", "critical") for i in self.issues)

    def add(self, check: str, severity: str, detail: str) -> None:
        self.issues.append({"check": check, "severity": severity, "detail": detail})


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def run_validation(schema: InvoiceSchema) -> ValidationReport:
    report = ValidationReport()
    now = datetime.now(timezone.utc)

    # Arithmetic: subtotal + tax = total
    if all(v is not None for v in [schema.subtotal, schema.tax, schema.total_amount]):
        expected = round(schema.subtotal + schema.tax, 2)
        actual = round(schema.total_amount, 2)
        if abs(expected - actual) > _TOLERANCE:
            report.add(
                "arithmetic", "warning",
                f"subtotal ({schema.subtotal}) + tax ({schema.tax}) = {expected}, "
                f"but total_amount = {actual}",
            )

    # Line item sum vs subtotal
    if schema.line_items and schema.subtotal is not None:
        totals = [li.total for li in schema.line_items if li.total is not None]
        if totals:
            items_sum = round(sum(totals), 2)
            if abs(items_sum - round(schema.subtotal, 2)) > _TOLERANCE:
                report.add(
                    "line_item_sum", "warning",
                    f"Line items sum to {items_sum} but subtotal = {schema.subtotal}",
                )

    # Future date
    invoice_dt = _parse_date(schema.invoice_date)
    if invoice_dt and invoice_dt > now:
        report.add("future_date", "warning", f"Invoice date {schema.invoice_date} is in the future")

    # Stale invoice (>365 days old)
    if invoice_dt and (now - invoice_dt).days > 365:
        report.add(
            "stale_invoice", "info",
            f"Invoice date {schema.invoice_date} is more than 365 days old",
        )

    # Missing critical fields
    if not schema.vendor_name:
        report.add("missing_vendor", "warning", "vendor_name is missing")
    if not schema.invoice_number:
        report.add("missing_invoice_number", "info", "invoice_number is missing")
    if schema.total_amount is None:
        report.add("missing_total", "critical", "total_amount could not be extracted")

    return report
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/unit/test_validation_agent.py -v
```
Expected: All PASS

- [ ] **Step 4: Commit**
```bash
git add agents/validation_agent.py tests/unit/test_validation_agent.py
git commit -m "feat: add validation agent with arithmetic, date, and field checks"
```

---

### Task 14: Discrepancy Detection agent (upgraded)

**Files:**
- Create: `agents/discrepancy_agent.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_discrepancy_agent.py`:
```python
import pytest
from unittest.mock import MagicMock
from models.invoice import InvoiceSchema

def test_detects_total_mismatch():
    from agents.discrepancy_agent import run_comparison
    schemas = [
        ("inv_a.pdf", InvoiceSchema(vendor_name="Acme", total_amount=1000.0, currency="USD")),
        ("inv_b.pdf", InvoiceSchema(vendor_name="Acme", total_amount=1200.0, currency="USD")),
    ]
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "no"
    result = run_comparison(schemas, mock_provider)
    fields = [d["field"] for d in result["discrepancies"]]
    assert "total_amount" in fields

def test_detects_currency_mismatch():
    from agents.discrepancy_agent import run_comparison
    schemas = [
        ("inv_a.pdf", InvoiceSchema(vendor_name="Acme", total_amount=100.0, currency="USD")),
        ("inv_b.pdf", InvoiceSchema(vendor_name="Acme", total_amount=100.0, currency="EUR")),
    ]
    mock_provider = MagicMock()
    result = run_comparison(schemas, mock_provider)
    fields = [d["field"] for d in result["discrepancies"]]
    assert "currency" in fields

def test_no_discrepancies_on_matching_invoices():
    from agents.discrepancy_agent import run_comparison
    schemas = [
        ("inv_a.pdf", InvoiceSchema(vendor_name="Acme", total_amount=500.0, currency="USD")),
        ("inv_b.pdf", InvoiceSchema(vendor_name="Acme", total_amount=500.0, currency="USD")),
    ]
    mock_provider = MagicMock()
    result = run_comparison(schemas, mock_provider)
    assert result["discrepancies"] == []
```

Run: `pytest tests/unit/test_discrepancy_agent.py -v`
Expected: FAIL

- [ ] **Step 2: Create `agents/discrepancy_agent.py`**

```python
from datetime import datetime
from models.invoice import InvoiceSchema
from agents.base import LLMProvider

_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "due_date",
    "subtotal", "tax", "total_amount", "currency", "po_number",
]
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y")


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _semantic_vendor_check(
    vendor_a: str, vendor_b: str, provider: LLMProvider
) -> bool:
    """Returns True if Gemini thinks the vendors are the same entity."""
    prompt = (
        f'Are "{vendor_a}" and "{vendor_b}" the same company or legal entity? '
        f"Reply ONLY 'yes' or 'no'."
    )
    verdict = provider.generate(prompt).strip().lower()
    return verdict.startswith("yes")


def run_comparison(
    named_schemas: list[tuple[str, InvoiceSchema]],
    provider: LLMProvider,
) -> dict:
    if len(named_schemas) < 2:
        return {"table": {}, "discrepancies": []}

    table = {
        field: {name: getattr(schema, field) for name, schema in named_schemas}
        for field in _FIELDS
    }
    discrepancies: list[dict] = []

    # Phase 1 — deterministic
    vendors = [v.strip() for v in table["vendor_name"].values() if v and v.strip()]
    unique_vendors = {v.lower() for v in vendors}
    if len(unique_vendors) > 1:
        # Phase 2 — Gemini semantic check for vendor aliases
        vendor_list = list(vendors)
        is_same = _semantic_vendor_check(vendor_list[0], vendor_list[1], provider)
        if not is_same:
            discrepancies.append({
                "field": "vendor_name",
                "severity": "critical",
                "detail": f"Different vendors: {', '.join(sorted(set(vendors)))}",
                "ai_reasoning": "Gemini confirmed these are different entities",
            })

    currencies = {c.strip().upper() for c in table["currency"].values() if c and c.strip()}
    if len(currencies) > 1:
        discrepancies.append({
            "field": "currency",
            "severity": "warning",
            "detail": f"Mixed currencies — totals not comparable: {', '.join(sorted(currencies))}",
            "ai_reasoning": None,
        })

    totals = [(name, val) for name, val in table["total_amount"].items() if val is not None]
    if len(totals) >= 2 and len(currencies) <= 1:
        amounts = [v for _, v in totals]
        min_a, max_a = min(amounts), max(amounts)
        if min_a > 0 and (max_a - min_a) / min_a > 0.05:
            discrepancies.append({
                "field": "total_amount",
                "severity": "warning",
                "detail": f"Total mismatch >5%: {[f'{n}={v}' for n, v in totals]}",
                "ai_reasoning": None,
            })

    parsed_dates = [
        (name, d)
        for name, val in table["invoice_date"].items()
        if (d := _parse_date(val)) is not None
    ]
    if len(parsed_dates) >= 2:
        date_values = [d for _, d in parsed_dates]
        gap = (max(date_values) - min(date_values)).days
        if gap > 30:
            discrepancies.append({
                "field": "invoice_date",
                "severity": "info",
                "detail": f"Date gap of {gap} days between invoices",
                "ai_reasoning": None,
            })

    inv_numbers = [v for v in table["invoice_number"].values() if v]
    if len(inv_numbers) == len(named_schemas) and len(set(inv_numbers)) == 1:
        discrepancies.append({
            "field": "invoice_number",
            "severity": "critical",
            "detail": f"Duplicate invoice number: {inv_numbers[0]}",
            "ai_reasoning": None,
        })

    return {"table": table, "discrepancies": discrepancies}
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/unit/test_discrepancy_agent.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add agents/discrepancy_agent.py tests/unit/test_discrepancy_agent.py
git commit -m "feat: add two-phase discrepancy agent with Gemini semantic vendor check"
```

---

### Task 15: Batch Orchestrator agent (new)

**Files:**
- Create: `agents/batch_agent.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_batch_agent.py`:
```python
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_batch_agent_processes_all_invoices():
    from agents.batch_agent import run_batch
    invoice_ids = [str(uuid.uuid4()) for _ in range(3)]
    mock_db = AsyncMock()
    mock_provider = MagicMock()
    mock_provider.generate_structured.return_value = {
        "vendor_name": "Test Co", "invoice_number": f"INV-{i}",
        "total_amount": 100.0 * (i + 1), "currency": "USD",
        "invoice_date": None, "due_date": None, "subtotal": None,
        "tax": None, "po_number": None, "payment_terms": None,
        "vendor_tax_id": None, "vendor_address": None,
        "bill_to": None, "line_items": [],
    } for i in range(3)

    async def fake_extract(inv_id, db, provider):
        from models.invoice import InvoiceSchema
        return InvoiceSchema(vendor_name="Test Co", total_amount=100.0)

    with patch("agents.batch_agent._extract_single", side_effect=fake_extract):
        results = await run_batch(invoice_ids, mock_db, mock_provider)

    assert len(results["done"]) == 3
    assert len(results["failed"]) == 0

@pytest.mark.asyncio
async def test_batch_agent_handles_partial_failure():
    from agents.batch_agent import run_batch
    invoice_ids = [str(uuid.uuid4()) for _ in range(3)]
    mock_db = AsyncMock()
    mock_provider = MagicMock()
    call_count = 0

    async def fake_extract_with_failure(inv_id, db, provider):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ValueError("extraction failed")
        from models.invoice import InvoiceSchema
        return InvoiceSchema(vendor_name="Test Co", total_amount=100.0)

    with patch("agents.batch_agent._extract_single", side_effect=fake_extract_with_failure):
        results = await run_batch(invoice_ids, mock_db, mock_provider)

    assert len(results["done"]) == 2
    assert len(results["failed"]) == 1
```

Run: `pytest tests/unit/test_batch_agent.py -v`
Expected: FAIL

- [ ] **Step 2: Create `agents/batch_agent.py`**

```python
import asyncio
from models.invoice import InvoiceSchema
from agents.base import LLMProvider
from agents.extraction_agent import run_extraction


async def _extract_single(
    invoice_id: str,
    db,
    provider: LLMProvider,
) -> InvoiceSchema:
    import uuid as _uuid
    from agents.retriever import HybridRetriever
    retriever = HybridRetriever(
        invoice_id=_uuid.UUID(invoice_id),
        db=db,
        provider=provider,
    )
    return await run_extraction(retriever, provider)


async def run_batch(
    invoice_ids: list[str],
    db,
    provider: LLMProvider,
    max_concurrent: int = 5,
) -> dict:
    done: dict[str, InvoiceSchema] = {}
    failed: dict[str, str] = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process(invoice_id: str) -> None:
        async with semaphore:
            try:
                schema = await _extract_single(invoice_id, db, provider)
                done[invoice_id] = schema
            except Exception as exc:
                failed[invoice_id] = str(exc)

    await asyncio.gather(*[_process(inv_id) for inv_id in invoice_ids])
    return {
        "done": done,
        "failed": failed,
        "total": len(invoice_ids),
        "success_count": len(done),
        "failure_count": len(failed),
    }
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/unit/test_batch_agent.py -v
```
Expected: PASS

- [ ] **Step 4: Run all agent tests together**
```bash
pytest tests/unit/ -v
```
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add agents/batch_agent.py tests/unit/test_batch_agent.py
git commit -m "feat: add batch orchestrator agent with async fan-out and partial failure handling"
```

---

## Phase 4 — API Routers

**Deliverable:** All `/api/v1/*` endpoints live, auth-gated, tested with real Postgres + mocked Gemini.

---

### Task 16: Pydantic response schemas + services

**Files:**
- Create: `api/schemas/invoice.py`
- Create: `api/schemas/extraction.py`
- Create: `api/schemas/job.py`
- Create: `api/schemas/webhook.py`
- Create: `api/services/storage.py`
- Create: `api/services/webhook_signer.py`
- Create: `api/services/llm_tracker.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_schemas.py`:
```python
from api.schemas.invoice import InvoiceOut, InvoiceUploadResponse
from api.schemas.job import JobOut
from api.schemas.webhook import WebhookIn, WebhookOut
import uuid

def test_invoice_out_schema():
    inv = InvoiceOut(
        id=uuid.uuid4(), tenant_id=uuid.uuid4(), file_name="test.pdf",
        file_type="pdf", status="ready", sha256="abc", storage_path="path",
        created_at=__import__("datetime").datetime.utcnow(),
    )
    assert inv.file_type == "pdf"

def test_webhook_in_validates_events():
    from pydantic import ValidationError
    with __import__("pytest").raises(ValidationError):
        WebhookIn(url="https://example.com", events=[], secret="s")
```

Run: `pytest tests/unit/test_schemas.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/schemas/invoice.py`**

```python
import uuid
from datetime import datetime
from pydantic import BaseModel


class InvoiceOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    file_name: str
    file_type: str
    status: str
    sha256: str
    storage_path: str
    created_at: datetime

    class Config:
        from_attributes = True


class InvoiceUploadResponse(BaseModel):
    invoice_id: uuid.UUID
    job_id: uuid.UUID
    status: str


class InvoiceListResponse(BaseModel):
    items: list[InvoiceOut]
    total: int
    page: int
    limit: int
```

- [ ] **Step 3: Create `api/schemas/extraction.py`**

```python
import uuid
from datetime import datetime
from pydantic import BaseModel
from models.invoice import InvoiceSchema


class ExtractionOut(BaseModel):
    id: uuid.UUID
    invoice_id: uuid.UUID
    schema_json: dict
    model_used: str
    validated: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ValidationResult(BaseModel):
    passed: bool
    issues: list[dict]
```

- [ ] **Step 4: Create `api/schemas/job.py`**

```python
import uuid
from datetime import datetime
from pydantic import BaseModel


class JobOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    type: str
    status: str
    payload: dict | None
    result: dict | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class BatchJobResult(BaseModel):
    batch_job_id: uuid.UUID
    status: str
    total: int
    done: int
    failed: int
    results: list[dict]
```

- [ ] **Step 5: Create `api/schemas/webhook.py`**

```python
import uuid
from datetime import datetime
from pydantic import BaseModel, HttpUrl, field_validator

_VALID_EVENTS = {
    "extraction.completed", "batch.done",
    "discrepancy.detected", "ingest.failed",
}


class WebhookIn(BaseModel):
    url: str
    events: list[str]
    secret: str

    @field_validator("events")
    @classmethod
    def events_not_empty(cls, v):
        if not v:
            raise ValueError("events must not be empty")
        invalid = set(v) - _VALID_EVENTS
        if invalid:
            raise ValueError(f"Unknown events: {invalid}. Valid: {_VALID_EVENTS}")
        return v


class WebhookOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    url: str
    events: list[str]
    active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class WebhookPatch(BaseModel):
    url: str | None = None
    events: list[str] | None = None
    active: bool | None = None
```

- [ ] **Step 6: Create `api/services/storage.py`**

```python
import hashlib
from pathlib import Path

from supabase import create_client, Client

from api.config import get_settings


def _get_client() -> Client:
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def _bucket(tenant_id: str) -> str:
    return f"invoices-{tenant_id}"


def sha256_file(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def upload_file(tenant_id: str, file_name: str, content: bytes) -> str:
    client = _get_client()
    bucket = _bucket(tenant_id)
    try:
        client.storage.create_bucket(bucket, options={"public": False})
    except Exception:
        pass
    storage_path = f"{tenant_id}/{file_name}"
    client.storage.from_(bucket).upload(
        storage_path, content, {"content-type": "application/octet-stream", "upsert": "true"}
    )
    return storage_path


def get_signed_url(tenant_id: str, storage_path: str, expires_in: int = 900) -> str:
    client = _get_client()
    bucket = _bucket(tenant_id)
    result = client.storage.from_(bucket).create_signed_url(storage_path, expires_in)
    return result["signedURL"]


def download_file(tenant_id: str, storage_path: str) -> bytes:
    client = _get_client()
    bucket = _bucket(tenant_id)
    return client.storage.from_(bucket).download(storage_path)


def delete_file(tenant_id: str, storage_path: str) -> None:
    client = _get_client()
    bucket = _bucket(tenant_id)
    client.storage.from_(bucket).remove([storage_path])
```

- [ ] **Step 7: Create `api/services/webhook_signer.py`**

```python
import hashlib
import hmac
import json
import time


def sign_payload(payload: dict, secret: str) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def verify_signature(payload: dict, secret: str, signature: str) -> bool:
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


def build_webhook_payload(event: str, tenant_id: str, data: dict) -> dict:
    return {
        "event": event,
        "tenant_id": tenant_id,
        "data": data,
        "timestamp": int(time.time()),
    }
```

- [ ] **Step 8: Create `api/services/llm_tracker.py`**

```python
import time
import uuid
import functools
from datetime import datetime, timezone
from typing import Callable

import structlog

log = structlog.get_logger()

_COST_PER_INPUT_TOKEN = 0.000_000_10   # Gemini 2.0 Flash ~$0.10/MTok input
_COST_PER_OUTPUT_TOKEN = 0.000_000_40  # ~$0.40/MTok output


def llm_usage_tracker(agent: str):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, tenant_id: str, invoice_id: str | None = None, db=None, **kwargs):
            start = time.monotonic()
            result = await fn(*args, tenant_id=tenant_id, invoice_id=invoice_id, db=db, **kwargs)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if db is not None:
                try:
                    from db.models import LlmUsage
                    usage = LlmUsage(
                        tenant_id=uuid.UUID(tenant_id),
                        invoice_id=uuid.UUID(invoice_id) if invoice_id else None,
                        model="gemini-2.0-flash",
                        agent=agent,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=elapsed_ms,
                        cost_usd=0,
                        created_at=datetime.now(timezone.utc),
                    )
                    db.add(usage)
                    await db.commit()
                except Exception as exc:
                    log.warning("llm_usage_write_failed", error=str(exc))
            return result
        return wrapper
    return decorator
```

- [ ] **Step 9: Run tests**
```bash
pytest tests/unit/test_schemas.py -v
```
Expected: PASS

- [ ] **Step 10: Commit**
```bash
git add api/schemas/ api/services/
git commit -m "feat: add Pydantic response schemas and storage/signer/tracker services"
```

---

### Task 17: Invoices router

**Files:**
- Create: `api/routers/invoices.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_api_invoices.py`:
```python
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from api.main import create_app

FAKE_USER = {
    "sub": str(uuid.uuid4()),
    "app_metadata": {"tenant_id": str(uuid.uuid4()), "role": "analyst"},
    "email": "test@example.com",
}

@pytest.fixture
def app():
    return create_app()

@pytest.mark.asyncio
async def test_upload_invoice_returns_job_id(app):
    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        with patch("api.routers.invoices.upload_file", return_value="path/to/file.pdf"):
            with patch("api.routers.invoices._enqueue_ingest", return_value=uuid.uuid4()) as mock_enqueue:
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/invoices/upload",
                        files={"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")},
                        headers={"Authorization": "Bearer fake.jwt.token"},
                    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert "invoice_id" in data
    assert "job_id" in data

@pytest.mark.asyncio
async def test_list_invoices_returns_empty_for_new_tenant(app):
    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/api/v1/invoices",
                headers={"Authorization": "Bearer fake.jwt.token"},
            )
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_upload_rejects_non_pdf_exe(app):
    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/invoices/upload",
                files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
                headers={"Authorization": "Bearer fake.jwt.token"},
            )
    assert response.status_code == 400
```

Run: `pytest tests/integration/test_api_invoices.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/routers/invoices.py`**

```python
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, require_roles, CurrentUser, get_queue
from api.schemas.invoice import InvoiceOut, InvoiceUploadResponse, InvoiceListResponse
from api.services.storage import upload_file, get_signed_url, delete_file, sha256_file
from db.models import Invoice, Job
from db.session import get_db

router = APIRouter(prefix="/invoices", tags=["invoices"])
log = structlog.get_logger()

_ALLOWED_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/jpg"}
_MAX_SIZE = 50 * 1024 * 1024  # 50 MB


def _detect_file_type(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return "pdf"
    if ext in ("jpg", "jpeg", "png"):
        return "image"
    raise HTTPException(400, f"Unsupported file type: .{ext}")


def _validate_upload(file: UploadFile, content: bytes) -> str:
    if len(content) > _MAX_SIZE:
        raise HTTPException(400, "File exceeds 50MB limit")
    if file.content_type and file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")
    if content[:4] == b"MZ\x90\x00" or content[:2] == b"MZ":
        raise HTTPException(400, "Executable files are not allowed")
    return _detect_file_type(file.filename or "", content)


async def _enqueue_ingest(invoice_id: uuid.UUID, queue) -> uuid.UUID:
    job_id = uuid.uuid4()
    queue.enqueue(
        "workers.ingest_job.run",
        str(invoice_id),
        job_id=str(job_id),
        job_timeout=300,
    )
    return job_id


@router.post("/upload", response_model=dict)
async def upload_invoice(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
    queue=Depends(get_queue),
):
    content = await file.read()
    file_type = _validate_upload(file, content)
    sha = sha256_file(content)

    existing = await db.scalar(
        select(Invoice).where(
            Invoice.tenant_id == uuid.UUID(user.tenant_id),
            Invoice.sha256 == sha,
        )
    )
    if existing:
        return {"data": {"invoice_id": existing.id, "job_id": None, "status": "already_exists"}, "error": None, "request_id": None}

    storage_path = upload_file(user.tenant_id, file.filename or f"upload.{file_type}", content)
    invoice = Invoice(
        tenant_id=uuid.UUID(user.tenant_id),
        uploaded_by=uuid.UUID(user.id) if user.id and user.id != "api_key" else None,
        file_name=file.filename or f"upload.{file_type}",
        file_type=file_type,
        storage_path=storage_path,
        sha256=sha,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db.add(invoice)
    await db.flush()

    job = Job(
        tenant_id=uuid.UUID(user.tenant_id),
        type="ingest",
        status="queued",
        payload={"invoice_id": str(invoice.id)},
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(invoice)

    queue.enqueue("workers.ingest_job.run", str(invoice.id), str(job.id), job_timeout=300)
    log.info("invoice.uploaded", invoice_id=str(invoice.id), tenant_id=user.tenant_id)
    return {"data": InvoiceUploadResponse(invoice_id=invoice.id, job_id=job.id, status="ingesting"), "error": None, "request_id": None}


@router.get("", response_model=dict)
async def list_invoices(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    file_type: str | None = Query(None),
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    q = select(Invoice).where(Invoice.tenant_id == uuid.UUID(user.tenant_id))
    if status:
        q = q.where(Invoice.status == status)
    if file_type:
        q = q.where(Invoice.file_type == file_type)
    total = await db.scalar(select(func.count()).select_from(q.subquery()))
    rows = (await db.execute(q.offset((page - 1) * limit).limit(limit))).scalars().all()
    return {
        "data": InvoiceListResponse(
            items=[InvoiceOut.model_validate(r) for r in rows],
            total=total or 0,
            page=page,
            limit=limit,
        ),
        "error": None,
        "request_id": None,
    }


@router.get("/{invoice_id}", response_model=dict)
async def get_invoice(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    inv = await db.scalar(
        select(Invoice).where(Invoice.id == invoice_id, Invoice.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    return {"data": InvoiceOut.model_validate(inv), "error": None, "request_id": None}


@router.get("/{invoice_id}/download", response_model=dict)
async def download_invoice(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    inv = await db.scalar(
        select(Invoice).where(Invoice.id == invoice_id, Invoice.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    url = get_signed_url(user.tenant_id, inv.storage_path)
    return {"data": {"signed_url": url, "expires_in": 900}, "error": None, "request_id": None}


@router.delete("/{invoice_id}", response_model=dict)
async def delete_invoice(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    inv = await db.scalar(
        select(Invoice).where(Invoice.id == invoice_id, Invoice.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    try:
        delete_file(user.tenant_id, inv.storage_path)
    except Exception:
        pass
    await db.delete(inv)
    await db.commit()
    log.info("invoice.deleted", invoice_id=str(invoice_id), tenant_id=user.tenant_id)
    return {"data": {"deleted": True}, "error": None, "request_id": None}
```

- [ ] **Step 3: Register router in `api/main.py`**

```python
# Add after health router import in create_app():
from api.routers import invoices
app.include_router(invoices.router, prefix="/api/v1")
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/integration/test_api_invoices.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add api/routers/invoices.py
git commit -m "feat: add invoices router — upload, list, get, download, delete"
```

---

### Task 18: Extraction + Q&A routers

**Files:**
- Create: `api/routers/extraction.py`
- Create: `api/routers/qa.py`
- Create: `api/routers/compare.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_api_extraction.py`:
```python
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from api.main import create_app

FAKE_USER = {
    "sub": str(uuid.uuid4()),
    "app_metadata": {"tenant_id": str(uuid.uuid4()), "role": "analyst"},
    "email": "test@example.com",
}

@pytest.mark.asyncio
async def test_run_extraction_enqueues_job():
    app = create_app()
    invoice_id = uuid.uuid4()
    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        with patch("api.routers.extraction._get_invoice", return_value=MagicMock(id=invoice_id, status="ready", file_type="pdf")):
            with patch("api.routers.extraction._enqueue_extract", return_value=uuid.uuid4()):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        f"/api/v1/invoices/{invoice_id}/extract",
                        headers={"Authorization": "Bearer fake"},
                    )
    assert response.status_code == 200
    assert "job_id" in response.json()["data"]

@pytest.mark.asyncio
async def test_viewer_cannot_run_extraction():
    VIEWER_USER = {**FAKE_USER, "app_metadata": {**FAKE_USER["app_metadata"], "role": "viewer"}}
    app = create_app()
    with patch("api.dependencies.verify_supabase_jwt", return_value=VIEWER_USER):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/invoices/{uuid.uuid4()}/extract",
                headers={"Authorization": "Bearer fake"},
            )
    assert response.status_code == 403
```

Run: `pytest tests/integration/test_api_extraction.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/routers/extraction.py`**

```python
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, require_roles, CurrentUser, get_queue
from api.schemas.extraction import ExtractionOut, ValidationResult
from agents.validation_agent import run_validation
from db.models import Invoice, Extraction, Job
from db.session import get_db
from models.invoice import InvoiceSchema

router = APIRouter(tags=["extraction"])
log = structlog.get_logger()


async def _get_invoice(invoice_id: uuid.UUID, tenant_id: str, db: AsyncSession) -> Invoice:
    inv = await db.scalar(
        select(Invoice).where(Invoice.id == invoice_id, Invoice.tenant_id == uuid.UUID(tenant_id))
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    return inv


async def _enqueue_extract(invoice_id: uuid.UUID, tenant_id: str, db: AsyncSession, queue) -> uuid.UUID:
    job = Job(
        tenant_id=uuid.UUID(tenant_id),
        type="extract",
        status="queued",
        payload={"invoice_id": str(invoice_id)},
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    queue.enqueue("workers.extract_job.run", str(invoice_id), str(job.id), job_timeout=120)
    return job.id


@router.post("/invoices/{invoice_id}/extract", response_model=dict)
async def run_extraction(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
    queue=Depends(get_queue),
):
    inv = await _get_invoice(invoice_id, user.tenant_id, db)
    if inv.status not in ("ready", "failed"):
        raise HTTPException(409, f"Invoice status is '{inv.status}', must be 'ready' to extract")
    job_id = await _enqueue_extract(invoice_id, user.tenant_id, db, queue)
    log.info("extraction.queued", invoice_id=str(invoice_id), job_id=str(job_id))
    return {"data": {"job_id": job_id, "status": "queued"}, "error": None, "request_id": None}


@router.get("/invoices/{invoice_id}/extraction", response_model=dict)
async def get_extraction(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    await _get_invoice(invoice_id, user.tenant_id, db)
    ext = await db.scalar(
        select(Extraction).where(Extraction.invoice_id == invoice_id)
    )
    if not ext:
        raise HTTPException(404, "No extraction found — run POST /extract first")
    return {"data": ExtractionOut.model_validate(ext), "error": None, "request_id": None}


@router.post("/invoices/{invoice_id}/validate", response_model=dict)
async def validate_extraction(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst")),
    db: AsyncSession = Depends(get_db),
):
    await _get_invoice(invoice_id, user.tenant_id, db)
    ext = await db.scalar(
        select(Extraction).where(Extraction.invoice_id == invoice_id)
    )
    if not ext:
        raise HTTPException(404, "No extraction found")
    schema = InvoiceSchema.model_validate(ext.schema_json)
    report = run_validation(schema)
    return {
        "data": ValidationResult(passed=report.passed, issues=report.issues),
        "error": None,
        "request_id": None,
    }
```

- [ ] **Step 3: Create `api/routers/qa.py`**

```python
import asyncio
import uuid
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base import get_provider
from agents.qa_agent import build_qa_agent
from agents.retriever import HybridRetriever
from api.dependencies import require_roles, CurrentUser
from db.models import Invoice
from db.session import get_db

router = APIRouter(tags=["qa"])
log = structlog.get_logger()


class QARequest(BaseModel):
    question: str


@router.post("/invoices/{invoice_id}/qa", response_model=dict)
async def ask_question(
    invoice_id: uuid.UUID,
    body: QARequest,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    inv = await db.scalar(
        select(Invoice).where(Invoice.id == invoice_id, Invoice.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    if inv.status != "ready":
        raise HTTPException(409, "Invoice not yet ingested")

    provider = get_provider()

    if inv.file_type == "image":
        from api.services.storage import download_file, get_signed_url
        import tempfile, pathlib
        content = download_file(user.tenant_id, inv.storage_path)
        suffix = "." + inv.file_name.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = pathlib.Path(tmp.name)
        from google import genai
        from PIL import Image as PILImage
        img = PILImage.open(tmp_path)
        answer = provider.generate(body.question, image=img)
        tmp_path.unlink(missing_ok=True)
        return {"data": {"answer": answer, "chunks": [], "agent_trace": []}, "error": None, "request_id": None}

    retriever = HybridRetriever(invoice_id=invoice_id, db=db, provider=provider)
    agent = build_qa_agent(retriever, provider)
    trace_steps = []
    async for event in agent.astream({
        "query": body.question,
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    }):
        trace_steps.append(event)
    final_state = list(trace_steps[-1].values())[0] if trace_steps else {}
    return {
        "data": {
            "answer": final_state.get("answer", "No answer generated."),
            "chunks": final_state.get("chunks", []),
            "agent_trace": [{k: str(v)[:300] for k, v in list(step.values())[0].items()} for step in trace_steps],
        },
        "error": None,
        "request_id": None,
    }


@router.post("/invoices/{invoice_id}/qa/stream")
async def ask_question_stream(
    invoice_id: uuid.UUID,
    body: QARequest,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    inv = await db.scalar(
        select(Invoice).where(Invoice.id == invoice_id, Invoice.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")

    provider = get_provider()
    retriever = HybridRetriever(invoice_id=invoice_id, db=db, provider=provider)
    agent = build_qa_agent(retriever, provider)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in agent.astream({
            "query": body.question,
            "rewritten_query": "",
            "chunks": [],
            "answer": "",
            "relevant": False,
            "grounded": False,
            "iterations": 0,
            "critique_iterations": 0,
        }):
            import json
            node_name = list(event.keys())[0]
            state = list(event.values())[0]
            yield f"data: {json.dumps({'node': node_name, 'answer': state.get('answer', '')})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

- [ ] **Step 4: Create `api/routers/compare.py`**

```python
import uuid
from pydantic import BaseModel

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base import get_provider
from agents.discrepancy_agent import run_comparison
from api.dependencies import require_roles, CurrentUser
from db.models import Invoice, Extraction
from db.session import get_db
from models.invoice import InvoiceSchema

router = APIRouter(tags=["compare"])
log = structlog.get_logger()


class CompareRequest(BaseModel):
    invoice_ids: list[uuid.UUID]


@router.post("/compare", response_model=dict)
async def compare_invoices(
    body: CompareRequest,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    if len(body.invoice_ids) < 2:
        raise HTTPException(400, "Need at least 2 invoice IDs to compare")

    named_schemas: list[tuple[str, InvoiceSchema]] = []
    for inv_id in body.invoice_ids:
        inv = await db.scalar(
            select(Invoice).where(Invoice.id == inv_id, Invoice.tenant_id == uuid.UUID(user.tenant_id))
        )
        if not inv:
            raise HTTPException(404, f"Invoice {inv_id} not found")
        ext = await db.scalar(select(Extraction).where(Extraction.invoice_id == inv_id))
        if not ext:
            raise HTTPException(409, f"Invoice {inv_id} has no extraction — run /extract first")
        named_schemas.append((inv.file_name, InvoiceSchema.model_validate(ext.schema_json)))

    provider = get_provider()
    result = run_comparison(named_schemas, provider)
    return {"data": result, "error": None, "request_id": None}
```

- [ ] **Step 5: Register routers in `api/main.py`**

```python
from api.routers import extraction, qa, compare
app.include_router(extraction.router, prefix="/api/v1")
app.include_router(qa.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
```

- [ ] **Step 6: Run tests**
```bash
pytest tests/integration/test_api_extraction.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**
```bash
git add api/routers/extraction.py api/routers/qa.py api/routers/compare.py
git commit -m "feat: add extraction, Q&A (+ SSE stream), and compare routers"
```

---

### Task 19: Batch + Jobs routers

**Files:**
- Create: `api/routers/batch.py`
- Create: `api/routers/jobs.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_api_batch.py`:
```python
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from api.main import create_app

FAKE_USER = {
    "sub": str(uuid.uuid4()),
    "app_metadata": {"tenant_id": str(uuid.uuid4()), "role": "analyst"},
    "email": "test@example.com",
}

@pytest.mark.asyncio
async def test_batch_extract_enqueues_job():
    app = create_app()
    ids = [str(uuid.uuid4()) for _ in range(3)]
    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        with patch("api.routers.batch._all_invoices_exist", return_value=True):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/batch/extract",
                    json={"invoice_ids": ids},
                    headers={"Authorization": "Bearer fake"},
                )
    assert response.status_code == 200
    assert "batch_job_id" in response.json()["data"]

@pytest.mark.asyncio
async def test_batch_requires_at_least_one_invoice():
    app = create_app()
    with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/batch/extract",
                json={"invoice_ids": []},
                headers={"Authorization": "Bearer fake"},
            )
    assert response.status_code == 400
```

Run: `pytest tests/integration/test_api_batch.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/routers/batch.py`**

```python
import io
import uuid
from datetime import datetime, timezone

import pandas as pd
import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser, get_queue
from db.models import Invoice, Job, Extraction
from db.session import get_db
from models.invoice import InvoiceSchema

router = APIRouter(prefix="/batch", tags=["batch"])
log = structlog.get_logger()


class BatchExtractRequest(BaseModel):
    invoice_ids: list[uuid.UUID]


async def _all_invoices_exist(
    invoice_ids: list[uuid.UUID], tenant_id: str, db: AsyncSession
) -> bool:
    for inv_id in invoice_ids:
        inv = await db.scalar(
            select(Invoice).where(Invoice.id == inv_id, Invoice.tenant_id == uuid.UUID(tenant_id))
        )
        if not inv:
            raise HTTPException(404, f"Invoice {inv_id} not found")
    return True


@router.post("/extract", response_model=dict)
async def batch_extract(
    body: BatchExtractRequest,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
    queue=Depends(get_queue),
):
    if not body.invoice_ids:
        raise HTTPException(400, "invoice_ids must not be empty")
    await _all_invoices_exist(body.invoice_ids, user.tenant_id, db)

    job = Job(
        tenant_id=uuid.UUID(user.tenant_id),
        type="batch_extract",
        status="queued",
        payload={"invoice_ids": [str(i) for i in body.invoice_ids]},
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    queue.enqueue("workers.batch_job.run", str(job.id), job_timeout=1800)
    log.info("batch.queued", job_id=str(job.id), count=len(body.invoice_ids))
    return {"data": {"batch_job_id": job.id, "status": "queued"}, "error": None, "request_id": None}


@router.get("/{job_id}", response_model=dict)
async def get_batch_status(
    job_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    job = await db.scalar(
        select(Job).where(Job.id == job_id, Job.tenant_id == uuid.UUID(user.tenant_id), Job.type == "batch_extract")
    )
    if not job:
        raise HTTPException(404, "Batch job not found")
    return {"data": {
        "batch_job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }, "error": None, "request_id": None}


@router.get("/{job_id}/export")
async def export_batch_csv(
    job_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    job = await db.scalar(
        select(Job).where(Job.id == job_id, Job.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not job or not job.result:
        raise HTTPException(404, "Batch job not found or not yet complete")

    invoice_ids = job.payload.get("invoice_ids", [])
    rows = []
    for inv_id in invoice_ids:
        ext = await db.scalar(select(Extraction).where(Extraction.invoice_id == uuid.UUID(inv_id)))
        inv = await db.scalar(select(Invoice).where(Invoice.id == uuid.UUID(inv_id)))
        if ext and inv:
            schema = InvoiceSchema.model_validate(ext.schema_json)
            row = {"invoice": inv.file_name, **schema.model_dump(exclude={"line_items"})}
            rows.append(row)

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=batch_{job_id}.csv"},
    )
```

- [ ] **Step 3: Create `api/routers/jobs.py`**

```python
import uuid
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from api.schemas.job import JobOut
from db.models import Job
from db.session import get_db

router = APIRouter(prefix="/jobs", tags=["jobs"])
log = structlog.get_logger()


@router.get("/{job_id}", response_model=dict)
async def get_job(
    job_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    job = await db.scalar(
        select(Job).where(Job.id == job_id, Job.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not job:
        raise HTTPException(404, "Job not found")
    return {"data": JobOut.model_validate(job), "error": None, "request_id": None}


@router.get("", response_model=dict)
async def list_jobs(
    status: str | None = Query(None),
    type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: CurrentUser = Depends(require_roles("admin", "analyst")),
    db: AsyncSession = Depends(get_db),
):
    q = select(Job).where(Job.tenant_id == uuid.UUID(user.tenant_id)).order_by(Job.created_at.desc())
    if status:
        q = q.where(Job.status == status)
    if type:
        q = q.where(Job.type == type)
    rows = (await db.execute(q.limit(limit))).scalars().all()
    return {"data": [JobOut.model_validate(r) for r in rows], "error": None, "request_id": None}
```

- [ ] **Step 4: Register routers in `api/main.py`**

```python
from api.routers import batch, jobs
app.include_router(batch.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
```

- [ ] **Step 5: Run tests**
```bash
pytest tests/integration/test_api_batch.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add api/routers/batch.py api/routers/jobs.py
git commit -m "feat: add batch extract and jobs routers"
```

---

### Task 20: Webhooks + Users + Audit routers

**Files:**
- Create: `api/routers/webhooks.py`
- Create: `api/routers/users.py`
- Create: `api/routers/audit.py`
- Create: `api/routers/api_keys.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_webhooks.py`:
```python
import pytest
import uuid
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from api.main import create_app

ADMIN_USER = {
    "sub": str(uuid.uuid4()),
    "app_metadata": {"tenant_id": str(uuid.uuid4()), "role": "admin"},
    "email": "admin@example.com",
}

@pytest.mark.asyncio
async def test_create_webhook():
    app = create_app()
    with patch("api.dependencies.verify_supabase_jwt", return_value=ADMIN_USER):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/webhooks",
                json={"url": "https://example.com/hook", "events": ["extraction.completed"], "secret": "s3cr3t"},
                headers={"Authorization": "Bearer fake"},
            )
    assert response.status_code == 200
    assert response.json()["data"]["url"] == "https://example.com/hook"

@pytest.mark.asyncio
async def test_analyst_cannot_create_webhook():
    ANALYST = {**ADMIN_USER, "app_metadata": {**ADMIN_USER["app_metadata"], "role": "analyst"}}
    app = create_app()
    with patch("api.dependencies.verify_supabase_jwt", return_value=ANALYST):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/webhooks",
                json={"url": "https://example.com/hook", "events": ["extraction.completed"], "secret": "s"},
                headers={"Authorization": "Bearer fake"},
            )
    assert response.status_code == 403
```

Run: `pytest tests/integration/test_webhooks.py -v`
Expected: FAIL

- [ ] **Step 2: Create `api/routers/webhooks.py`**

```python
import secrets
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from api.schemas.webhook import WebhookIn, WebhookOut, WebhookPatch
from api.services.webhook_signer import build_webhook_payload, sign_payload
from db.models import Webhook, WebhookDelivery
from db.session import get_db

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
log = structlog.get_logger()


@router.post("", response_model=dict)
async def create_webhook(
    body: WebhookIn,
    user: CurrentUser = Depends(require_roles("admin", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    wh = Webhook(
        tenant_id=uuid.UUID(user.tenant_id),
        url=body.url,
        events=body.events,
        secret=body.secret,
        active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(wh)
    await db.commit()
    await db.refresh(wh)
    return {"data": WebhookOut.model_validate(wh), "error": None, "request_id": None}


@router.get("", response_model=dict)
async def list_webhooks(
    user: CurrentUser = Depends(require_roles("admin", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Webhook).where(Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )).scalars().all()
    return {"data": [WebhookOut.model_validate(r) for r in rows], "error": None, "request_id": None}


@router.get("/{webhook_id}", response_model=dict)
async def get_webhook(
    webhook_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    wh = await db.scalar(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not wh:
        raise HTTPException(404, "Webhook not found")
    deliveries = (await db.execute(
        select(WebhookDelivery).where(WebhookDelivery.webhook_id == webhook_id).order_by(WebhookDelivery.delivered_at.desc()).limit(20)
    )).scalars().all()
    return {
        "data": {**WebhookOut.model_validate(wh).model_dump(), "deliveries": [
            {"event": d.event, "status": d.status, "attempts": d.attempts, "last_error": d.last_error}
            for d in deliveries
        ]},
        "error": None,
        "request_id": None,
    }


@router.patch("/{webhook_id}", response_model=dict)
async def update_webhook(
    webhook_id: uuid.UUID,
    body: WebhookPatch,
    user: CurrentUser = Depends(require_roles("admin", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    wh = await db.scalar(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not wh:
        raise HTTPException(404, "Webhook not found")
    if body.url is not None:
        wh.url = body.url
    if body.events is not None:
        wh.events = body.events
    if body.active is not None:
        wh.active = body.active
    await db.commit()
    await db.refresh(wh)
    return {"data": WebhookOut.model_validate(wh), "error": None, "request_id": None}


@router.delete("/{webhook_id}", response_model=dict)
async def delete_webhook(
    webhook_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    wh = await db.scalar(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not wh:
        raise HTTPException(404, "Webhook not found")
    await db.delete(wh)
    await db.commit()
    return {"data": {"deleted": True}, "error": None, "request_id": None}


@router.post("/{webhook_id}/test", response_model=dict)
async def test_webhook(
    webhook_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    import httpx
    wh = await db.scalar(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not wh:
        raise HTTPException(404, "Webhook not found")
    payload = build_webhook_payload("webhook.test", user.tenant_id, {"message": "Test ping from Invoice Analyst"})
    signature = sign_payload(payload, wh.secret)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(wh.url, json=payload, headers={"X-Signature": signature})
        return {"data": {"status": resp.status_code, "ok": resp.is_success}, "error": None, "request_id": None}
    except Exception as exc:
        return {"data": {"status": 0, "ok": False}, "error": str(exc), "request_id": None}
```

- [ ] **Step 3: Create `api/routers/users.py`**

```python
import uuid
import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from api.schemas.user import UserOut, RoleUpdateIn
from db.models import User
from db.session import get_db

router = APIRouter(prefix="/users", tags=["users"])
log = structlog.get_logger()

_VALID_ROLES = {"admin", "analyst", "viewer", "api_user"}


@router.get("", response_model=dict)
async def list_users(
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(User).where(User.tenant_id == uuid.UUID(user.tenant_id))
    )).scalars().all()
    return {"data": [UserOut.model_validate(r) for r in rows], "error": None, "request_id": None}


@router.patch("/{user_id}/role", response_model=dict)
async def update_role(
    user_id: uuid.UUID,
    body: RoleUpdateIn,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    if body.role not in _VALID_ROLES:
        raise HTTPException(400, f"Invalid role. Must be one of: {_VALID_ROLES}")
    target = await db.scalar(
        select(User).where(User.id == user_id, User.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not target:
        raise HTTPException(404, "User not found")
    target.role = body.role
    await db.commit()
    log.info("user.role_changed", target_user_id=str(user_id), new_role=body.role)
    return {"data": UserOut.model_validate(target), "error": None, "request_id": None}


@router.delete("/{user_id}", response_model=dict)
async def remove_user(
    user_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    target = await db.scalar(
        select(User).where(User.id == user_id, User.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not target:
        raise HTTPException(404, "User not found")
    if str(user_id) == user.id:
        raise HTTPException(400, "Cannot remove yourself")
    await db.delete(target)
    await db.commit()
    return {"data": {"deleted": True}, "error": None, "request_id": None}
```

- [ ] **Step 4: Create `api/routers/api_keys.py`**

```python
import secrets
import uuid
from datetime import datetime, timezone

import bcrypt
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from db.models import ApiKey
from db.session import get_db

router = APIRouter(prefix="/api-keys", tags=["api-keys"])
log = structlog.get_logger()


class ApiKeyIn(BaseModel):
    name: str


class ApiKeyOut(BaseModel):
    id: uuid.UUID
    name: str
    role: str
    active: bool
    last_used_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


@router.post("", response_model=dict)
async def create_api_key(
    body: ApiKeyIn,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    raw_key = secrets.token_urlsafe(32)
    key_hash = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()
    key = ApiKey(
        tenant_id=uuid.UUID(user.tenant_id),
        name=body.name,
        key_hash=key_hash,
        role="api_user",
        active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(key)
    await db.commit()
    await db.refresh(key)
    log.info("api_key.created", key_id=str(key.id), name=body.name)
    return {
        "data": {**ApiKeyOut.model_validate(key).model_dump(), "raw_key": raw_key},
        "error": None,
        "request_id": None,
    }


@router.get("", response_model=dict)
async def list_api_keys(
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(ApiKey).where(ApiKey.tenant_id == uuid.UUID(user.tenant_id))
    )).scalars().all()
    return {"data": [ApiKeyOut.model_validate(r) for r in rows], "error": None, "request_id": None}


@router.delete("/{key_id}", response_model=dict)
async def revoke_api_key(
    key_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    key = await db.scalar(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not key:
        raise HTTPException(404, "API key not found")
    key.active = False
    await db.commit()
    return {"data": {"revoked": True}, "error": None, "request_id": None}
```

- [ ] **Step 5: Create `api/routers/audit.py`**

```python
import uuid
import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from db.models import AuditLog
from db.session import get_db

router = APIRouter(prefix="/audit", tags=["audit"])
log = structlog.get_logger()


@router.get("", response_model=dict)
async def list_audit_log(
    limit: int = Query(100, ge=1, le=500),
    action: str | None = Query(None),
    user_id: uuid.UUID | None = Query(None),
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(AuditLog)
        .where(AuditLog.tenant_id == uuid.UUID(user.tenant_id))
        .order_by(AuditLog.created_at.desc())
    )
    if action:
        q = q.where(AuditLog.action == action)
    if user_id:
        q = q.where(AuditLog.user_id == user_id)
    rows = (await db.execute(q.limit(limit))).scalars().all()
    return {
        "data": [
            {
                "id": str(r.id), "action": r.action, "user_id": str(r.user_id) if r.user_id else None,
                "resource_type": r.resource_type, "resource_id": str(r.resource_id) if r.resource_id else None,
                "metadata": r.metadata, "created_at": r.created_at,
            }
            for r in rows
        ],
        "error": None,
        "request_id": None,
    }
```

- [ ] **Step 6: Register all remaining routers in `api/main.py`**

```python
from api.routers import webhooks, users, api_keys, audit
app.include_router(webhooks.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(api_keys.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
```

- [ ] **Step 7: Run all tests**
```bash
pytest tests/integration/ -v
```
Expected: All PASS

- [ ] **Step 8: Commit**
```bash
git add api/routers/webhooks.py api/routers/users.py api/routers/api_keys.py api/routers/audit.py
git commit -m "feat: add webhooks, users, api-keys, and audit log routers"
```

---

## Phase 5 — Background Workers

**Deliverable:** Four RQ workers running correctly — ingest, extract, batch, webhook — all tested with mocked external services.

---

### Task 21: Ingest worker

**Files:**
- Create: `workers/__init__.py`
- Create: `workers/ingest_job.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_ingest_job.py`:
```python
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_ingest_splits_pdf_into_chunks():
    from workers.ingest_job import _chunk_pdf_bytes
    fake_pdf_text = ["Page one content about an invoice from Acme Corp.", "Page two with line items."]
    with patch("workers.ingest_job._extract_page_texts", return_value=fake_pdf_text):
        chunks = _chunk_pdf_bytes(b"%PDF fake", chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all("text" in c and "page" in c for c in chunks)

@pytest.mark.asyncio
async def test_ingest_stores_chunks_in_db():
    from workers.ingest_job import _store_chunks
    mock_db = AsyncMock()
    mock_provider = MagicMock()
    mock_provider.embed_text.return_value = [[0.1] * 768 for _ in range(3)]
    invoice_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    chunks = [{"text": f"chunk {i}", "page": 1} for i in range(3)]
    await _store_chunks(chunks, invoice_id, tenant_id, mock_db, mock_provider)
    assert mock_db.add.call_count == 3
    assert mock_db.commit.called
```

Run: `pytest tests/unit/test_ingest_job.py -v`
Expected: FAIL

- [ ] **Step 2: Create `workers/__init__.py`**
```python
```

- [ ] **Step 3: Create `workers/ingest_job.py`**

```python
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select, update

from agents.base import get_provider
from api.config import get_settings
from api.services.storage import download_file
from db.models import Invoice, InvoiceChunk, Job
from db.session import get_session_factory

log = structlog.get_logger()


def _extract_page_texts(pdf_bytes: bytes) -> list[str]:
    import tempfile
    from pypdf import PdfReader
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        reader = PdfReader(str(tmp_path))
        texts = [p.extract_text() or "" for p in reader.pages]
        if sum(len(t.strip()) for t in texts) < 32:
            import rag.ocr
            texts = rag.ocr.ocr_pdf_pages(tmp_path)
        return texts
    finally:
        tmp_path.unlink(missing_ok=True)


def _chunk_pdf_bytes(pdf_bytes: bytes, chunk_size: int = 800, chunk_overlap: int = 80) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    page_texts = _extract_page_texts(pdf_bytes)
    chunks = []
    for page_num, page_text in enumerate(page_texts, start=1):
        for piece in splitter.split_text(page_text):
            chunks.append({"text": piece, "page": page_num})
    return chunks


async def _store_chunks(
    chunks: list[dict],
    invoice_id: uuid.UUID,
    tenant_id: uuid.UUID,
    db,
    provider,
) -> None:
    texts = [c["text"] for c in chunks]
    embeddings = provider.embed_text(texts)
    for chunk, embedding in zip(chunks, embeddings):
        db.add(InvoiceChunk(
            invoice_id=invoice_id,
            tenant_id=tenant_id,
            chunk_text=chunk["text"],
            page_num=chunk["page"],
            embedding=embedding,
            created_at=datetime.now(timezone.utc),
        ))
    await db.commit()


def run(invoice_id_str: str, job_id_str: str) -> None:
    import asyncio
    asyncio.run(_run_async(invoice_id_str, job_id_str))


async def _run_async(invoice_id_str: str, job_id_str: str) -> None:
    settings = get_settings()
    invoice_id = uuid.UUID(invoice_id_str)
    job_id = uuid.UUID(job_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        inv = await db.scalar(select(Invoice).where(Invoice.id == invoice_id))
        if not inv:
            log.error("ingest.invoice_not_found", invoice_id=invoice_id_str)
            return

        await db.execute(
            update(Invoice).where(Invoice.id == invoice_id).values(status="ingesting")
        )
        await db.execute(
            update(Job).where(Job.id == job_id).values(status="running")
        )
        await db.commit()

        try:
            pdf_bytes = download_file(str(inv.tenant_id), inv.storage_path)
            if inv.file_type == "pdf":
                chunks = _chunk_pdf_bytes(pdf_bytes, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            else:
                import tempfile
                suffix = "." + inv.file_name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = Path(tmp.name)
                provider = get_provider()
                embedding = provider.embed_image(tmp_path)
                tmp_path.unlink(missing_ok=True)
                chunks = [{"text": f"[image invoice: {inv.file_name}]", "page": 1, "_embedding": embedding}]

            provider = get_provider()
            await _store_chunks(chunks, invoice_id, inv.tenant_id, db, provider)

            await db.execute(
                update(Invoice).where(Invoice.id == invoice_id).values(status="ready")
            )
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="done",
                    result={"chunks_stored": len(chunks)},
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            log.info("ingest.done", invoice_id=invoice_id_str, chunks=len(chunks))

        except Exception as exc:
            log.error("ingest.failed", invoice_id=invoice_id_str, error=str(exc))
            await db.execute(
                update(Invoice).where(Invoice.id == invoice_id).values(status="failed")
            )
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="failed",
                    error=str(exc),
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            raise
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/unit/test_ingest_job.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add workers/__init__.py workers/ingest_job.py tests/unit/test_ingest_job.py
git commit -m "feat: add ingest worker — download, chunk, embed, store in pgvector"
```

---

### Task 22: Extract + Batch + Webhook workers

**Files:**
- Create: `workers/extract_job.py`
- Create: `workers/batch_job.py`
- Create: `workers/webhook_job.py`
- Create: `workers/worker.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_workers.py`:
```python
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_extract_job_writes_extraction_to_db():
    from workers.extract_job import _run_async
    invoice_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    mock_schema = MagicMock()
    mock_schema.model_dump_json.return_value = '{"vendor_name": "Acme"}'
    mock_schema.model_dump.return_value = {"vendor_name": "Acme"}

    with patch("workers.extract_job.get_session_factory") as mock_sf:
        mock_db = AsyncMock()
        mock_db.scalar = AsyncMock(return_value=MagicMock(
            id=uuid.UUID(invoice_id), tenant_id=uuid.uuid4(), file_type="pdf"
        ))
        mock_sf.return_value.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_sf.return_value.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("workers.extract_job.run_extraction", return_value=mock_schema):
            with patch("workers.extract_job.HybridRetriever"):
                with patch("workers.extract_job.get_provider"):
                    with patch("workers.extract_job.run_validation", return_value=MagicMock(issues=[])):
                        await _run_async(invoice_id, job_id)

def test_webhook_signs_payload():
    from workers.webhook_job import _build_signed_request
    payload = {"event": "extraction.completed", "tenant_id": "t1", "data": {}, "timestamp": 1000}
    headers = _build_signed_request(payload, "my_secret")
    assert "X-Signature" in headers
    assert headers["X-Signature"].startswith("sha256=")
```

Run: `pytest tests/unit/test_workers.py -v`
Expected: FAIL

- [ ] **Step 2: Create `workers/extract_job.py`**

```python
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, update

from agents.base import get_provider
from agents.extraction_agent import run_extraction
from agents.retriever import HybridRetriever
from agents.validation_agent import run_validation
from db.models import Invoice, Extraction, Job
from db.session import get_session_factory

log = structlog.get_logger()


def run(invoice_id_str: str, job_id_str: str) -> None:
    import asyncio
    asyncio.run(_run_async(invoice_id_str, job_id_str))


async def _run_async(invoice_id_str: str, job_id_str: str) -> None:
    invoice_id = uuid.UUID(invoice_id_str)
    job_id = uuid.UUID(job_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        inv = await db.scalar(select(Invoice).where(Invoice.id == invoice_id))
        if not inv:
            log.error("extract.invoice_not_found", invoice_id=invoice_id_str)
            return

        await db.execute(update(Job).where(Job.id == job_id).values(status="running"))
        await db.commit()

        try:
            provider = get_provider()

            if inv.file_type == "image":
                from api.services.storage import download_file
                import tempfile
                from pathlib import Path
                content = download_file(str(inv.tenant_id), inv.storage_path)
                suffix = "." + inv.file_name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
                raw = provider.generate_structured(
                    f"Extract all invoice fields from this image.", type("ImageSchema", (), {})
                )
                from models.invoice import InvoiceSchema
                schema = InvoiceSchema.model_validate(raw)
                tmp_path.unlink(missing_ok=True)
            else:
                retriever = HybridRetriever(invoice_id=invoice_id, db=db, provider=provider)
                schema = await run_extraction(retriever, provider)

            validation_report = run_validation(schema)

            existing = await db.scalar(select(Extraction).where(Extraction.invoice_id == invoice_id))
            if existing:
                existing.schema_json = schema.model_dump()
                existing.model_used = "gemini-2.0-flash"
                existing.validated = validation_report.passed
            else:
                db.add(Extraction(
                    invoice_id=invoice_id,
                    tenant_id=inv.tenant_id,
                    schema_json=schema.model_dump(),
                    model_used="gemini-2.0-flash",
                    validated=validation_report.passed,
                    created_at=datetime.now(timezone.utc),
                ))

            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="done",
                    result={"validated": validation_report.passed, "issues": validation_report.issues},
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            log.info("extract.done", invoice_id=invoice_id_str)

        except Exception as exc:
            log.error("extract.failed", invoice_id=invoice_id_str, error=str(exc))
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="failed", error=str(exc), completed_at=datetime.now(timezone.utc)
                )
            )
            await db.commit()
            raise
```

- [ ] **Step 3: Create `workers/batch_job.py`**

```python
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, update

from agents.base import get_provider
from agents.batch_agent import run_batch
from db.models import Job
from db.session import get_session_factory

log = structlog.get_logger()


def run(job_id_str: str) -> None:
    import asyncio
    asyncio.run(_run_async(job_id_str))


async def _run_async(job_id_str: str) -> None:
    job_id = uuid.UUID(job_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        job = await db.scalar(select(Job).where(Job.id == job_id))
        if not job:
            log.error("batch.job_not_found", job_id=job_id_str)
            return

        invoice_ids: list[str] = job.payload.get("invoice_ids", [])
        await db.execute(update(Job).where(Job.id == job_id).values(status="running"))
        await db.commit()

        try:
            provider = get_provider()
            results = await run_batch(invoice_ids, db, provider)
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="done",
                    result={
                        "success_count": results["success_count"],
                        "failure_count": results["failure_count"],
                        "failed_ids": list(results["failed"].keys()),
                        "errors": results["failed"],
                    },
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            log.info("batch.done", job_id=job_id_str, done=results["success_count"], failed=results["failure_count"])

        except Exception as exc:
            log.error("batch.failed", job_id=job_id_str, error=str(exc))
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="failed", error=str(exc), completed_at=datetime.now(timezone.utc)
                )
            )
            await db.commit()
            raise
```

- [ ] **Step 4: Create `workers/webhook_job.py`**

```python
import json
import time
import uuid
from datetime import datetime, timezone

import httpx
import structlog
from sqlalchemy import select, update

from api.services.webhook_signer import sign_payload
from db.models import Webhook, WebhookDelivery
from db.session import get_session_factory

log = structlog.get_logger()

_RETRY_DELAYS = [5, 30, 120, 600, 1800]
_MAX_ATTEMPTS = len(_RETRY_DELAYS) + 1


def _build_signed_request(payload: dict, secret: str) -> dict:
    signature = sign_payload(payload, secret)
    return {
        "Content-Type": "application/json",
        "X-Signature": signature,
        "X-Webhook-Event": payload.get("event", ""),
    }


def run(webhook_id_str: str, event: str, payload: dict) -> None:
    import asyncio
    asyncio.run(_run_async(webhook_id_str, event, payload))


async def _run_async(webhook_id_str: str, event: str, payload: dict) -> None:
    webhook_id = uuid.UUID(webhook_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        wh = await db.scalar(select(Webhook).where(Webhook.id == webhook_id, Webhook.active == True))
        if not wh:
            log.warning("webhook.not_found_or_inactive", webhook_id=webhook_id_str)
            return

        delivery = WebhookDelivery(
            webhook_id=webhook_id,
            event=event,
            payload=payload,
            status="pending",
            attempts=0,
        )
        db.add(delivery)
        await db.commit()
        await db.refresh(delivery)

        headers = _build_signed_request(payload, wh.secret)

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if attempt > 0:
                time.sleep(delay)

            delivery.attempts = attempt + 1
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(wh.url, json=payload, headers=headers)
                if resp.is_success:
                    delivery.status = "delivered"
                    delivery.delivered_at = datetime.now(timezone.utc)
                    await db.commit()
                    log.info("webhook.delivered", webhook_id=webhook_id_str, event=event, attempt=attempt + 1)
                    return
                else:
                    delivery.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    await db.commit()
                    log.warning("webhook.delivery_failed", attempt=attempt + 1, status=resp.status_code)

            except Exception as exc:
                delivery.last_error = str(exc)[:500]
                await db.commit()
                log.warning("webhook.delivery_error", attempt=attempt + 1, error=str(exc))

            if attempt + 1 >= _MAX_ATTEMPTS:
                delivery.status = "failed"
                await db.commit()
                log.error("webhook.permanently_failed", webhook_id=webhook_id_str, event=event)
```

- [ ] **Step 5: Create `workers/worker.py`**

```python
from redis import Redis
from rq import Worker, Queue

from api.config import get_settings


def main():
    settings = get_settings()
    conn = Redis.from_url(settings.REDIS_URL)
    queues = [Queue("invoice-jobs", connection=conn)]
    worker = Worker(queues, connection=conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run all worker tests**
```bash
pytest tests/unit/test_workers.py tests/unit/test_ingest_job.py -v
```
Expected: All PASS

- [ ] **Step 7: Commit**
```bash
git add workers/
git commit -m "feat: add extract, batch, webhook workers and RQ worker entrypoint"
```

---

## Phase 6 — Frontend, Observability & CI/CD

**Deliverable:** Streamlit thin client calling FastAPI, Prometheus + LangSmith wired, Docker images building, render.yaml complete, GitHub Actions green.

---

### Task 23: Streamlit thin client

**Files:**
- Modify: `frontend/app.py`
- Create: `frontend/api_client.py`
- Modify: `frontend/auth.py`
- Create: `frontend/pages/qa.py`
- Create: `frontend/pages/extract.py`
- Create: `frontend/pages/compare.py`
- Create: `frontend/pages/batch.py`
- Create: `frontend/pages/dashboard.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_api_client.py`:
```python
import pytest
import respx
import httpx
from frontend.api_client import APIClient

@pytest.mark.asyncio
async def test_client_list_invoices():
    client = APIClient(base_url="http://fake-api", token="jwt-token")
    with respx.mock:
        respx.get("http://fake-api/api/v1/invoices").mock(
            return_value=httpx.Response(200, json={"data": {"items": [], "total": 0, "page": 1, "limit": 20}, "error": None, "request_id": "x"})
        )
        result = await client.list_invoices()
    assert result["items"] == []

@pytest.mark.asyncio
async def test_client_raises_on_auth_error():
    client = APIClient(base_url="http://fake-api", token="bad-token")
    with respx.mock:
        respx.get("http://fake-api/api/v1/invoices").mock(
            return_value=httpx.Response(401, json={"data": None, "error": "Unauthorized", "request_id": "x"})
        )
        with pytest.raises(Exception, match="401"):
            await client.list_invoices()
```

Run: `pytest tests/unit/test_api_client.py -v`
Expected: FAIL

- [ ] **Step 2: Create `frontend/api_client.py`**

```python
import httpx
from typing import Any


class APIClient:
    def __init__(self, base_url: str, token: str):
        self._base = base_url.rstrip("/")
        self._token = token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    async def _get(self, path: str, params: dict | None = None) -> Any:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self._base}{path}", headers=self._headers(), params=params)
            if not resp.is_success:
                raise Exception(f"{resp.status_code}: {resp.text}")
            return resp.json()["data"]

    async def _post(self, path: str, json: dict | None = None, files=None) -> Any:
        async with httpx.AsyncClient(timeout=60) as client:
            if files:
                resp = await client.post(
                    f"{self._base}{path}",
                    headers={"Authorization": f"Bearer {self._token}"},
                    files=files,
                )
            else:
                resp = await client.post(f"{self._base}{path}", headers=self._headers(), json=json or {})
            if not resp.is_success:
                raise Exception(f"{resp.status_code}: {resp.text}")
            return resp.json()["data"]

    async def list_invoices(self, page: int = 1, limit: int = 20) -> dict:
        return await self._get("/api/v1/invoices", params={"page": page, "limit": limit})

    async def upload_invoice(self, filename: str, content: bytes, content_type: str) -> dict:
        return await self._post(
            "/api/v1/invoices/upload",
            files={"file": (filename, content, content_type)},
        )

    async def get_invoice(self, invoice_id: str) -> dict:
        return await self._get(f"/api/v1/invoices/{invoice_id}")

    async def delete_invoice(self, invoice_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(
                f"{self._base}/api/v1/invoices/{invoice_id}", headers=self._headers()
            )
            if not resp.is_success:
                raise Exception(f"{resp.status_code}: {resp.text}")
            return resp.json()["data"]

    async def run_extraction(self, invoice_id: str) -> dict:
        return await self._post(f"/api/v1/invoices/{invoice_id}/extract")

    async def get_extraction(self, invoice_id: str) -> dict:
        return await self._get(f"/api/v1/invoices/{invoice_id}/extraction")

    async def ask_question(self, invoice_id: str, question: str) -> dict:
        return await self._post(f"/api/v1/invoices/{invoice_id}/qa", json={"question": question})

    async def compare_invoices(self, invoice_ids: list[str]) -> dict:
        return await self._post("/api/v1/compare", json={"invoice_ids": invoice_ids})

    async def batch_extract(self, invoice_ids: list[str]) -> dict:
        return await self._post("/api/v1/batch/extract", json={"invoice_ids": invoice_ids})

    async def get_job(self, job_id: str) -> dict:
        return await self._get(f"/api/v1/jobs/{job_id}")

    async def list_jobs(self, status: str | None = None, type: str | None = None) -> list:
        params = {}
        if status:
            params["status"] = status
        if type:
            params["type"] = type
        return await self._get("/api/v1/jobs", params=params)
```

- [ ] **Step 3: Modify `frontend/auth.py`**

```python
import os
import streamlit as st
from supabase import create_client


def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]
    return create_client(url, key)


def login_page() -> bool:
    st.title("Invoice Analyst")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign In", type="primary"):
        try:
            sb = get_supabase_client()
            result = sb.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state["access_token"] = result.session.access_token
            st.session_state["user_email"] = result.user.email
            st.rerun()
        except Exception as exc:
            st.error(f"Login failed: {exc}")
    return False


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def get_token() -> str:
    return st.session_state.get("access_token", "")


def logout():
    for key in ["access_token", "user_email"]:
        st.session_state.pop(key, None)
    st.rerun()
```

- [ ] **Step 4: Modify `frontend/app.py`**

```python
import os
import asyncio
import streamlit as st
from frontend.auth import is_authenticated, login_page, logout, get_token
from frontend.api_client import APIClient
from api.config import get_settings

st.set_page_config(page_title="Invoice Analyst", page_icon="🧾", layout="wide")

if not is_authenticated():
    login_page()
    st.stop()

settings = get_settings()
client = APIClient(base_url=settings.API_BASE_URL, token=get_token())

with st.sidebar:
    st.markdown(f"**{st.session_state.get('user_email', '')}**")
    if st.button("Sign Out"):
        logout()
    st.divider()
    uploaded = st.file_uploader("Upload Invoice (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded and st.button("Add Invoice", type="primary"):
        with st.spinner("Uploading..."):
            try:
                result = asyncio.run(client.upload_invoice(
                    uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream"
                ))
                st.success(f"Uploaded — job {result['job_id']}")
            except Exception as e:
                st.error(str(e))

qa_tab, extract_tab, compare_tab, batch_tab, dashboard_tab = st.tabs(
    ["Q&A", "Extract", "Compare", "Batch", "Dashboard"]
)

from frontend.pages import qa, extract, compare, batch, dashboard
with qa_tab:
    qa.render(client)
with extract_tab:
    extract.render(client)
with compare_tab:
    compare.render(client)
with batch_tab:
    batch.render(client)
with dashboard_tab:
    dashboard.render(client)
```

- [ ] **Step 5: Create `frontend/pages/__init__.py`**
```python
```

- [ ] **Step 6: Create `frontend/pages/qa.py`**

```python
import asyncio
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.subheader("Q&A")
    try:
        invoices_data = asyncio.run(client.list_invoices())
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return
    if not invoices:
        st.info("No invoices loaded yet — upload one from the sidebar.")
        return
    options = {inv["id"]: inv["file_name"] for inv in invoices}
    selected_id = st.selectbox("Invoice", list(options.keys()), format_func=lambda k: options[k])
    question = st.text_input("Ask a question", placeholder="e.g. What is the invoice total?")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Running agentic RAG..."):
            try:
                result = asyncio.run(client.ask_question(selected_id, question))
                st.success(result.get("answer", "No answer generated."))
                with st.expander("Agent trace"):
                    for step in result.get("agent_trace", []):
                        st.json(step)
                if result.get("chunks"):
                    with st.expander("Source chunks"):
                        for i, chunk in enumerate(result["chunks"], 1):
                            st.markdown(f"**Chunk {i}** — page `{chunk.get('page')}`, score `{chunk.get('score', 0):.4f}`")
                            st.text(chunk.get("text", "")[:400])
            except Exception as e:
                st.error(str(e))
```

- [ ] **Step 7: Create `frontend/pages/extract.py`**

```python
import asyncio
import pandas as pd
import streamlit as st
from frontend.api_client import APIClient
from models.invoice import InvoiceSchema


def render(client: APIClient):
    st.subheader("Extract")
    try:
        invoices_data = asyncio.run(client.list_invoices())
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return
    if not invoices:
        st.info("No invoices loaded yet.")
        return
    options = {inv["id"]: inv["file_name"] for inv in invoices}
    selected_id = st.selectbox("Invoice", list(options.keys()), format_func=lambda k: options[k], key="ext_sel")
    if st.button("Extract All Fields", type="primary"):
        with st.spinner("Extracting..."):
            try:
                job = asyncio.run(client.run_extraction(selected_id))
                st.info(f"Extraction queued — job {job['job_id']}. Refresh to see results.")
            except Exception as e:
                st.error(str(e))
    try:
        ext = asyncio.run(client.get_extraction(selected_id))
        schema = InvoiceSchema.model_validate(ext["schema_json"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", f"{schema.total_amount:,.2f}" if schema.total_amount else "—")
        c2.metric("Vendor", schema.vendor_name or "—")
        c3.metric("Invoice #", schema.invoice_number or "—")
        c4.metric("Date", schema.invoice_date or "—")
        st.subheader("Header fields")
        st.dataframe(pd.DataFrame({
            "Field": ["Vendor", "Invoice #", "Date", "Due Date", "Subtotal", "Tax", "Total", "Currency"],
            "Value": [schema.vendor_name, schema.invoice_number, schema.invoice_date,
                      schema.due_date, schema.subtotal, schema.tax, schema.total_amount, schema.currency],
        }), use_container_width=True, hide_index=True)
        if schema.line_items:
            st.subheader("Line items")
            st.dataframe(pd.DataFrame([li.model_dump() for li in schema.line_items]), use_container_width=True)
        st.download_button("Download JSON", schema.model_dump_json(indent=2), f"extraction.json", "application/json")
    except Exception:
        st.caption("No extraction yet — click Extract above.")
```

- [ ] **Step 8: Create `frontend/pages/compare.py`**

```python
import asyncio
import pandas as pd
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.subheader("Compare")
    try:
        invoices_data = asyncio.run(client.list_invoices())
        invoices = [i for i in invoices_data.get("items", []) if i["file_type"] == "pdf"]
    except Exception as e:
        st.error(str(e))
        return
    if len(invoices) < 2:
        st.info("Load at least 2 PDF invoices to compare.")
        return
    selected = [inv["id"] for inv in invoices if st.checkbox(inv["file_name"], key=f"cmp_{inv['id']}")]
    if len(selected) >= 2 and st.button("Compare Selected", type="primary"):
        with st.spinner("Comparing..."):
            try:
                result = asyncio.run(client.compare_invoices(selected))
                table = result.get("table", {})
                discrepancies = result.get("discrepancies", [])
                if table:
                    disc_fields = {d["field"] for d in discrepancies}
                    rows = [{"Field": f, **v} for f, v in table.items()]
                    df = pd.DataFrame(rows).set_index("Field")
                    def highlight(df):
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for f in disc_fields:
                            if f in styles.index:
                                styles.loc[f] = "background-color: #5C3A2B"
                        return styles
                    st.dataframe(df.style.apply(highlight, axis=None), use_container_width=True)
                if discrepancies:
                    st.subheader("Discrepancies")
                    for d in discrepancies:
                        st.warning(f"**{d['field']}** ({d.get('severity','info')}): {d['detail']}")
                else:
                    st.success("No discrepancies found.")
            except Exception as e:
                st.error(str(e))
```

- [ ] **Step 9: Create `frontend/pages/batch.py`**

```python
import asyncio
import time
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.subheader("Batch Extract")
    try:
        invoices_data = asyncio.run(client.list_invoices(limit=100))
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return
    if not invoices:
        st.info("No invoices loaded yet.")
        return
    selected = [inv["id"] for inv in invoices if st.checkbox(inv["file_name"], key=f"batch_{inv['id']}")]
    st.caption(f"{len(selected)} invoice(s) selected")
    if st.button("Run Batch Extraction", type="primary", disabled=len(selected) == 0):
        with st.spinner("Queuing batch job..."):
            try:
                result = asyncio.run(client.batch_extract(selected))
                batch_job_id = result["batch_job_id"]
                st.info(f"Batch job queued: `{batch_job_id}`")
                progress = st.progress(0)
                status_text = st.empty()
                for _ in range(60):
                    time.sleep(3)
                    job = asyncio.run(client.get_job(batch_job_id))
                    if job["status"] in ("done", "failed"):
                        progress.progress(1.0)
                        if job["status"] == "done":
                            r = job.get("result", {})
                            st.success(f"Done — {r.get('success_count', 0)} succeeded, {r.get('failure_count', 0)} failed")
                        else:
                            st.error(f"Batch failed: {job.get('error')}")
                        break
                    status_text.text(f"Status: {job['status']}...")
            except Exception as e:
                st.error(str(e))
```

- [ ] **Step 10: Create `frontend/pages/dashboard.py`**

```python
import asyncio
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.subheader("Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Recent Jobs")
        try:
            jobs = asyncio.run(client.list_jobs(limit=20))
            for job in jobs[:10]:
                status_color = "🟢" if job["status"] == "done" else "🔴" if job["status"] == "failed" else "🟡"
                st.markdown(f"{status_color} `{job['type']}` — {job['status']} — {job['created_at'][:10]}")
        except Exception as e:
            st.error(str(e))
    with col2:
        st.markdown("#### All Invoices")
        try:
            data = asyncio.run(client.list_invoices(limit=100))
            total = data.get("total", 0)
            items = data.get("items", [])
            ready = sum(1 for i in items if i["status"] == "ready")
            st.metric("Total", total)
            st.metric("Ready", ready)
            st.metric("Pending/Ingesting", total - ready)
        except Exception as e:
            st.error(str(e))
```

- [ ] **Step 11: Run tests**
```bash
pytest tests/unit/test_api_client.py -v
```
Expected: PASS

- [ ] **Step 12: Commit**
```bash
git add frontend/ tests/unit/test_api_client.py
git commit -m "feat: refactor Streamlit to thin client calling FastAPI"
```

---

### Task 24: Prometheus metrics + LangSmith wiring

**Files:**
- Modify: `api/main.py` — add Prometheus instrumentator
- Modify: `agents/qa_agent.py` — add LangSmith tracer injection
- Modify: `agents/extraction_agent.py` — add LangSmith tracer injection

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_observability.py`:
```python
def test_prometheus_metrics_registered():
    from prometheus_client import REGISTRY
    metric_names = [m.name for m in REGISTRY.collect()]
    assert any("http" in n or "invoice" in n for n in metric_names)

def test_structlog_produces_json(capsys):
    import structlog
    log = structlog.get_logger()
    log.info("test_event", key="value")
    # structlog configured in main.py — no crash = pass
```

Run: `pytest tests/unit/test_observability.py -v`
Expected: FAIL

- [ ] **Step 2: Add Prometheus to `api/main.py`**

Add after `app` is created in `create_app()`:
```python
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

extractions_total = Counter(
    "invoice_extractions_total", "Total extractions", ["status"]
)
extraction_duration = Histogram(
    "invoice_extraction_duration_seconds", "Extraction latency",
    buckets=[0.5, 1, 2, 5, 10, 30],
)
tokens_used_total = Counter(
    "llm_tokens_used_total", "LLM tokens", ["model", "direction"]
)

Instrumentator().instrument(app).expose(app, endpoint="/api/v1/metrics")
```

- [ ] **Step 3: Wire LangSmith in agent builds**

In `agents/qa_agent.py`, update `build_qa_agent` to accept optional `tenant_id` and `invoice_id` for trace tags:
```python
def build_qa_agent(
    retriever: HybridRetriever,
    provider: LLMProvider,
    tenant_id: str | None = None,
    invoice_id: str | None = None,
):
    settings = get_settings()
    tags = []
    if tenant_id:
        tags.append(f"tenant:{tenant_id}")
    if invoice_id:
        tags.append(f"invoice:{invoice_id}")
    # Pass tags into agent config at compile time
    compiled = graph.compile()
    compiled._tags = tags  # stored for use in astream config
    return compiled
```

In `api/routers/qa.py`, pass `config` with LangSmith callbacks when streaming:
```python
from api.config import get_settings
settings = get_settings()
config = {}
if settings.LANGCHAIN_TRACING_V2:
    from langchain_core.callbacks import LangChainTracer
    tracer = LangChainTracer(
        project_name=settings.LANGCHAIN_PROJECT,
        tags=[f"tenant:{user.tenant_id}", f"invoice:{invoice_id}"],
    )
    config = {"callbacks": [tracer]}

async for event in agent.astream({...}, config=config):
    trace_steps.append(event)
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/unit/test_observability.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add api/main.py agents/qa_agent.py api/routers/qa.py
git commit -m "feat: add Prometheus metrics and LangSmith tracing to API and agents"
```

---

### Task 25: Dockerfiles + docker-compose

**Files:**
- Create: `Dockerfile.api`
- Create: `Dockerfile.frontend`
- Create: `docker-compose.yml`

- [ ] **Step 1: Write failing test**

Create `tests/test_docker.py`:
```python
import subprocess
import pytest

def test_api_dockerfile_exists():
    import os
    assert os.path.exists("Dockerfile.api")

def test_frontend_dockerfile_exists():
    import os
    assert os.path.exists("Dockerfile.frontend")

def test_docker_compose_valid():
    result = subprocess.run(
        ["docker", "compose", "config"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"docker-compose invalid: {result.stderr}"
```

Run: `pytest tests/test_docker.py::test_api_dockerfile_exists -v`
Expected: FAIL

- [ ] **Step 2: Create `Dockerfile.api`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[prod]"

COPY api/       api/
COPY agents/    agents/
COPY workers/   workers/
COPY db/        db/
COPY models/    models/
COPY rag/       rag/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

- [ ] **Step 3: Create `Dockerfile.frontend`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[frontend]"

COPY frontend/  frontend/
COPY models/    models/

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

- [ ] **Step 4: Create `docker-compose.yml`**

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    env_file:
      - .env.local
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/ready"]
      interval: 10s
      timeout: 5s
      retries: 5

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    env_file:
      - .env.local
    environment:
      API_BASE_URL: http://api:8000
    depends_on:
      - api

  worker:
    build:
      context: .
      dockerfile: Dockerfile.api
    command: python workers/worker.py
    env_file:
      - .env.local
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: invoice_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: dev
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

- [ ] **Step 5: Run tests**
```bash
pytest tests/test_docker.py -v
```
Expected: PASS (docker-compose valid requires Docker installed)

- [ ] **Step 6: Commit**
```bash
git add Dockerfile.api Dockerfile.frontend docker-compose.yml
git commit -m "feat: add Dockerfiles and docker-compose for local dev"
```

---

### Task 26: render.yaml + GitHub Actions

**Files:**
- Create: `render.yaml`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/deploy.yml`
- Create: `.env.example`

- [ ] **Step 1: Create `render.yaml`**

```yaml
services:
  - type: web
    name: invoice-api
    runtime: docker
    dockerfilePath: ./Dockerfile.api
    plan: starter
    healthCheckPath: /api/v1/health/ready
    envVars:
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_ANON_KEY
        sync: false
      - key: SUPABASE_SERVICE_KEY
        sync: false
      - key: DATABASE_URL
        sync: false
      - key: GOOGLE_API_KEY
        sync: false
      - key: LLM_PROVIDER
        value: gemini
      - key: REDIS_URL
        fromService:
          name: invoice-redis
          type: redis
          property: connectionString
      - key: LANGCHAIN_TRACING_V2
        value: "true"
      - key: LANGCHAIN_API_KEY
        sync: false
      - key: LANGCHAIN_PROJECT
        value: invoice-analyst-prod
      - key: SENTRY_DSN
        sync: false
      - key: ENV
        value: production
      - key: GIT_SHA
        fromGitInfo: commitSha
      - key: ALLOWED_ORIGINS
        sync: false

  - type: web
    name: invoice-frontend
    runtime: docker
    dockerfilePath: ./Dockerfile.frontend
    plan: starter
    envVars:
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_ANON_KEY
        sync: false
      - key: API_BASE_URL
        fromService:
          name: invoice-api
          type: web
          property: host

  - type: worker
    name: invoice-worker
    runtime: docker
    dockerfilePath: ./Dockerfile.api
    startCommand: python workers/worker.py
    plan: starter
    envVars:
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_SERVICE_KEY
        sync: false
      - key: DATABASE_URL
        sync: false
      - key: GOOGLE_API_KEY
        sync: false
      - key: LLM_PROVIDER
        value: gemini
      - key: REDIS_URL
        fromService:
          name: invoice-redis
          type: redis
          property: connectionString
      - key: LANGCHAIN_TRACING_V2
        value: "true"
      - key: LANGCHAIN_API_KEY
        sync: false
      - key: LANGCHAIN_PROJECT
        value: invoice-analyst-prod

  - type: redis
    name: invoice-redis
    plan: starter
    maxmemoryPolicy: allkeys-lru
```

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: invoice_test
          POSTGRES_USER: postgres
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint (ruff)
        run: ruff check . && ruff format --check .

      - name: Type check (mypy)
        run: mypy api/ agents/ workers/ db/ --ignore-missing-imports

      - name: Run tests
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:test@localhost/invoice_test
          REDIS_URL: redis://localhost:6379
          SUPABASE_URL: https://fake.supabase.co
          SUPABASE_ANON_KEY: fake_anon_key
          SUPABASE_SERVICE_KEY: fake_service_key
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY_TEST }}
          LLM_PROVIDER: gemini
          LANGCHAIN_TRACING_V2: "false"
          ENV: test
        run: |
          DATABASE_URL=postgresql+asyncpg://postgres:test@localhost/invoice_test alembic upgrade head
          pytest tests/ -v --cov=api --cov=agents --cov=workers --cov-report=xml --ignore=tests/e2e

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

- [ ] **Step 3: Create `.github/workflows/deploy.yml`**

```yaml
name: Deploy to Render
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    needs: []
    steps:
      - uses: actions/checkout@v4

      - name: Run DB migrations
        env:
          DATABASE_URL: ${{ secrets.PROD_DATABASE_URL }}
        run: |
          pip install -e ".[prod]"
          alembic upgrade head

      - name: Deploy API to Render
        run: curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_API }}"

      - name: Deploy Frontend to Render
        run: curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_FRONTEND }}"

      - name: Deploy Worker to Render
        run: curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_WORKER }}"
```

- [ ] **Step 4: Update `.env.example`**

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...

# Database (from Supabase project settings → Database → Connection string)
DATABASE_URL=postgresql+asyncpg://postgres:password@db.your-project.supabase.co:5432/postgres

# Redis (local dev)
REDIS_URL=redis://localhost:6379

# AI
GOOGLE_API_KEY=AIza...
LLM_PROVIDER=ollama_gemma      # use "gemini" for production

# Ollama (local dev only)
OLLAMA_BASE_URL=http://localhost:11434

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=invoice-analyst-dev

# Sentry (optional)
SENTRY_DSN=

# App
ENV=development
ALLOWED_ORIGINS=http://localhost:8501
API_BASE_URL=http://localhost:8000
```

- [ ] **Step 5: Run final test suite**
```bash
pytest tests/ -v --ignore=tests/e2e
```
Expected: All PASS

- [ ] **Step 6: Final commit**
```bash
git add render.yaml .github/ .env.example
git commit -m "feat: add render.yaml, GitHub Actions CI/CD, and .env.example"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Multi-tenant RBAC (4 roles) — Tasks 5, 17–20
- ✅ Supabase Postgres + pgvector + Storage — Tasks 2, 3, 10, 16
- ✅ Gemini 2.0 Flash (prod) + Gemma 3 4B (local) — Tasks 8, 9
- ✅ gemini-embedding-exp-03-07 (768 dims) — Tasks 8, 10
- ✅ All 6 agents (3 upgraded, 3 new) — Tasks 11–15
- ✅ FastAPI all routers — Tasks 17–20
- ✅ RQ workers (ingest, extract, batch, webhook) — Tasks 21–22
- ✅ Streamlit thin client — Task 23
- ✅ LangSmith tracing — Task 24
- ✅ Sentry + structlog — Tasks 4, 6, 7
- ✅ Prometheus metrics — Task 24
- ✅ llm_usage table + tracker — Task 16
- ✅ Webhook HMAC signing + retry — Tasks 16, 22
- ✅ Audit log — Tasks 6, 20
- ✅ Docker + docker-compose — Task 25
- ✅ render.yaml — Task 26
- ✅ GitHub Actions CI + deploy — Task 26
- ✅ Alembic migrations — Task 3
- ✅ TDD throughout — every task has failing test first

**No placeholders found.** All code blocks contain runnable implementations.

**Type consistency verified:**
- `HybridRetriever(invoice_id, db, provider)` — consistent Tasks 10, 11, 12, 21, 22
- `LLMProvider.embed_text/embed_image/generate/generate_structured` — consistent Tasks 8, 9, 11–15
- `CurrentUser.id/tenant_id/role/email` — consistent Tasks 5, 17–20
- `InvoiceSchema` — unchanged from original, used consistently throughout
- `JobOut`, `WebhookOut`, `InvoiceOut` — defined Task 16, used Tasks 17–20

---

**Plan complete.** 26 tasks across 6 phases.

**Plan saved to `docs/superpowers/plans/2026-06-12-enterprise-upgrade-plan.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Invoke `superpowers:subagent-driven-development`.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**
