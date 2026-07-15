# Conversational Cross-Invoice Chat + Liquid Glass Frontend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persisted multi-turn chat over all of a tenant's invoices, with a routed LangGraph agent (aggregate / detail / image-vision paths) and an Apple liquid-glass Streamlit frontend.

**Architecture:** New `conversations`/`conversation_messages` tables (Alembic 0002). `HybridRetriever` gains a tenant-wide mode. A new `agents/chat_agent.py` LangGraph condenses follow-ups against history, routes questions to structured-extraction answering, cross-invoice hybrid RAG, or Gemini/Ollama vision. New `/api/v1/chat/*` REST endpoints persist turns. Streamlit gets a Chat tab and a shared glassmorphic theme.

**Tech Stack:** FastAPI, SQLAlchemy async + Alembic, LangGraph, pgvector, rank-bm25, Streamlit, Gemini / Ollama providers.

**Spec:** `docs/superpowers/specs/2026-07-10-conversational-chat-design.md`

**Working directory:** `C:\Users\amanb\invoice-extractor` (all commands run from here). Run tests with `python -m pytest`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `db/models.py` | Modify | Add `Conversation`, `ConversationMessage` |
| `db/migrations/versions/0002_conversations.py` | Create | Migration for the two tables |
| `agents/base.py` | Modify | Add `generate_with_image` to `LLMProvider` protocol |
| `agents/providers/gemini.py` | Modify | Implement `generate_with_image` (native multimodal) |
| `agents/providers/ollama_gemma.py` | Modify | Implement `generate_with_image` (images param + graceful degrade) |
| `agents/retriever.py` | Modify | Tenant-wide retrieval mode with file_name metadata |
| `agents/chat_agent.py` | Create | Routed conversational LangGraph agent |
| `api/schemas/chat.py` | Create | Request/response models |
| `api/routers/chat.py` | Create | Conversation CRUD + message endpoint |
| `api/main.py` | Modify | Register chat router |
| `frontend/api_client.py` | Modify | Chat client methods + `_delete` helper |
| `frontend/theme.py` | Create | Liquid-glass CSS, `inject_theme()` |
| `frontend/pages/chat.py` | Create | Chat page UI |
| `frontend/app.py` | Modify | Inject theme, add Chat tab |
| `README.md` | Modify | Document the feature |

Tests: `tests/unit/test_chat_models.py`, `tests/unit/test_provider_vision.py`, `tests/unit/test_retriever_tenant.py`, `tests/unit/test_chat_agent.py`, `tests/unit/test_chat_schemas.py`, `tests/integration/test_api_chat.py`, `tests/unit/test_api_client_chat.py`, `tests/unit/test_theme.py`.

---

### Task 1: Conversation DB models + migration

**Files:**
- Modify: `db/models.py` (append at end)
- Create: `db/migrations/versions/0002_conversations.py`
- Test: `tests/unit/test_chat_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_chat_models.py`:

```python
def test_conversation_model_columns():
    from db.models import Conversation
    assert Conversation.__tablename__ == "conversations"
    cols = {c.name for c in Conversation.__table__.columns}
    assert {"id", "tenant_id", "user_id", "title", "created_at", "updated_at"} <= cols


def test_conversation_message_model_columns():
    from db.models import ConversationMessage
    assert ConversationMessage.__tablename__ == "conversation_messages"
    cols = {c.name for c in ConversationMessage.__table__.columns}
    assert {"id", "conversation_id", "tenant_id", "role", "content", "metadata", "created_at"} <= cols


def test_conversation_cascades_messages():
    from db.models import Conversation
    rel = Conversation.__mapper__.relationships["messages"]
    assert rel.cascade.delete_orphan
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_chat_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'Conversation'`

- [ ] **Step 3: Add models**

Append to `db/models.py`:

```python
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    title = Column(String(255), nullable=False, server_default="New conversation")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    messages = relationship(
        "ConversationMessage", back_populates="conversation",
        cascade="all, delete-orphan", order_by="ConversationMessage.created_at",
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(10), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    meta = Column("metadata", JSONB, nullable=True)  # sources, route, trace
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    conversation = relationship("Conversation", back_populates="messages")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_chat_models.py -v`
Expected: 3 PASS

- [ ] **Step 5: Create migration**

Create `db/migrations/versions/0002_conversations.py`:

```python
"""conversations for multi-turn chat

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-10
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column("title", sa.String(255), nullable=False, server_default="New conversation"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_table(
        "conversation_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("conversation_messages")
    op.drop_table("conversations")
```

- [ ] **Step 6: Verify migration chain**

Run: `python -m alembic history`
Expected output includes: `0001 -> 0002 (head), conversations for multi-turn chat`

- [ ] **Step 7: Commit**

```bash
git add db/models.py db/migrations/versions/0002_conversations.py tests/unit/test_chat_models.py
git commit -m "feat(chat): conversation and message models with migration 0002"
```

---

### Task 2: Provider vision support (`generate_with_image`)

**Files:**
- Modify: `agents/base.py`
- Modify: `agents/providers/gemini.py`
- Modify: `agents/providers/ollama_gemma.py`
- Test: `tests/unit/test_provider_vision.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_provider_vision.py`:

```python
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

_TEST_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
    "REDIS_URL": "redis://localhost:6379",
    "SUPABASE_URL": "https://fake.supabase.co",
    "SUPABASE_ANON_KEY": "fake_anon",
    "SUPABASE_SERVICE_KEY": "fake_service",
    "SUPABASE_JWT_SECRET": "test_secret",
    "GOOGLE_API_KEY": "fake_google_key",
}


def test_provider_protocol_includes_vision():
    from agents.base import LLMProvider
    assert hasattr(LLMProvider, "generate_with_image")


class _FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "The total is $42"}


def _fake_client_factory(captured, post_exc=None):
    class _FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None):
            if post_exc:
                raise post_exc
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()

    return _FakeClient


def test_ollama_generate_with_image_sends_image_payload(tmp_path):
    from agents.providers.ollama_gemma import OllamaGemmaProvider
    img = tmp_path / "invoice.png"
    img.write_bytes(b"fake-png-bytes")
    provider = OllamaGemmaProvider(base_url="http://localhost:11434", model="gemma3:4b")
    captured = {}
    with patch("agents.providers.ollama_gemma.httpx.Client", _fake_client_factory(captured)):
        out = provider.generate_with_image("what is the total?", img)
    assert out == "The total is $42"
    assert captured["json"]["images"], "expected base64 image in payload"


def test_ollama_generate_with_image_degrades_gracefully(tmp_path):
    from agents.providers.ollama_gemma import OllamaGemmaProvider
    img = tmp_path / "invoice.png"
    img.write_bytes(b"fake-png-bytes")
    provider = OllamaGemmaProvider(base_url="http://localhost:11434", model="gemma3:4b")
    exc = httpx.HTTPStatusError("boom", request=MagicMock(), response=MagicMock())
    with patch("agents.providers.ollama_gemma.httpx.Client", _fake_client_factory({}, post_exc=exc)):
        out = provider.generate_with_image("what is the total?", img)
    assert "image" in out.lower()  # explains, does not raise


def test_gemini_generate_with_image_calls_multimodal(tmp_path):
    img = tmp_path / "invoice.png"
    img.write_bytes(b"fake-png-bytes")
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        with patch("agents.providers.gemini.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = MagicMock(text="Total: $99")
            mock_client_cls.return_value = mock_client
            from agents.providers.gemini import GeminiProvider
            provider = GeminiProvider()
            with patch("PIL.Image.open", return_value=MagicMock()):
                out = provider.generate_with_image("what is the total?", img)
    assert out == "Total: $99"
    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert len(contents) == 2  # prompt + image
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_provider_vision.py -v`
Expected: FAIL — protocol lacks `generate_with_image`, providers raise `AttributeError`

- [ ] **Step 3: Implement**

In `agents/base.py`, add to the `LLMProvider` protocol (after `generate_structured`):

```python
    def generate_with_image(self, prompt: str, image_path: Path, system: str | None = None) -> str:
        ...
```

In `agents/providers/gemini.py`, add to `GeminiProvider`:

```python
    def generate_with_image(self, prompt: str, image_path: Path, system: str | None = None) -> str:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        config = types.GenerateContentConfig(system_instruction=system) if system else None
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, img],
            config=config,
        )
        return response.text or ""
```

In `agents/providers/ollama_gemma.py`, add to `OllamaGemmaProvider`:

```python
    def generate_with_image(self, prompt: str, image_path: Path, system: str | None = None) -> str:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            return self.generate(prompt, system=system, images=[img_b64])
        except httpx.HTTPError as exc:
            log.warning("ollama_vision_failed", error=str(exc))
            return (
                "I couldn't analyse this image with the configured local model. "
                "Image questions work best with the Gemini provider."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_provider_vision.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add agents/base.py agents/providers/gemini.py agents/providers/ollama_gemma.py tests/unit/test_provider_vision.py
git commit -m "feat(chat): add generate_with_image to LLM providers"
```

---

### Task 3: Tenant-wide retriever mode

**Files:**
- Modify: `agents/retriever.py`
- Test: `tests/unit/test_retriever_tenant.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_retriever_tenant.py`:

```python
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


def _rows(tenant=True):
    base = [
        {"id": uuid.uuid4(), "chunk_text": "Total due $500 from Acme Corp", "page_num": 1},
        {"id": uuid.uuid4(), "chunk_text": "Total due $900 from Globex Inc", "page_num": 1},
    ]
    if tenant:
        for i, r in enumerate(base):
            r["invoice_id"] = uuid.uuid4()
            r["file_name"] = ["acme.pdf", "globex.pdf"][i]
    return base


@pytest.mark.asyncio
async def test_tenant_mode_filters_by_tenant_and_returns_file_names():
    from agents.retriever import HybridRetriever
    tenant_id = uuid.uuid4()
    rows = _rows()
    queries = []

    async def fake_execute(query, params):
        queries.append((str(query), params))
        if "embedding" in str(query):
            return FakeResult([{**r, "similarity": 0.9} for r in rows])
        return FakeResult(rows)

    db = MagicMock()
    db.execute = AsyncMock(side_effect=fake_execute)
    provider = MagicMock()
    provider.embed_text = MagicMock(return_value=[[0.0] * 768])

    r = HybridRetriever(invoice_id=None, db=db, provider=provider, num_results=2, tenant_id=tenant_id)
    out = await r.retrieve("total due")

    assert len(out) == 2
    assert all("file_name" in c and "invoice_id" in c for c in out)
    corpus_sql, corpus_params = queries[0]
    assert "tenant_id" in corpus_sql
    assert "file_type" in corpus_sql  # image placeholders excluded
    assert corpus_params["tid"] == str(tenant_id)


@pytest.mark.asyncio
async def test_single_invoice_mode_unchanged():
    from agents.retriever import HybridRetriever
    rows = _rows(tenant=False)
    queries = []

    async def fake_execute(query, params):
        queries.append((str(query), params))
        if "embedding" in str(query):
            return FakeResult([{**r, "similarity": 0.9} for r in rows])
        return FakeResult(rows)

    db = MagicMock()
    db.execute = AsyncMock(side_effect=fake_execute)
    provider = MagicMock()
    provider.embed_text = MagicMock(return_value=[[0.0] * 768])
    inv_id = uuid.uuid4()

    r = HybridRetriever(invoice_id=inv_id, db=db, provider=provider, num_results=2)
    out = await r.retrieve("total due")
    assert len(out) == 2
    assert "inv_id" in queries[0][1]


def test_requires_invoice_or_tenant():
    from agents.retriever import HybridRetriever
    with pytest.raises(ValueError):
        HybridRetriever(invoice_id=None, db=MagicMock(), provider=MagicMock(), tenant_id=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_retriever_tenant.py -v`
Expected: FAIL — `TypeError` (no `tenant_id` kwarg) / no `ValueError` raised

- [ ] **Step 3: Implement tenant mode**

In `agents/retriever.py`, replace `__init__`, `_load_corpus`, and `_dense_retrieve`; update `_bm25_retrieve` and `retrieve` to carry the metadata:

```python
class HybridRetriever:
    def __init__(
        self,
        invoice_id: uuid.UUID | None = None,
        db: AsyncSession = None,
        provider: LLMProvider = None,
        num_results: int = 4,
        tenant_id: uuid.UUID | None = None,
    ):
        if invoice_id is None and tenant_id is None:
            raise ValueError("HybridRetriever requires an invoice_id or a tenant_id")
        self._invoice_id = invoice_id
        self._tenant_id = tenant_id
        self._db = db
        self._provider = provider
        self._num_results = num_results
        self._corpus: list[dict] | None = None
        self._bm25: BM25Okapi | None = None
```

`_load_corpus` — branch on mode; tenant mode joins `invoices` for `file_name` and excludes image placeholder chunks:

```python
    async def _load_corpus(self) -> list[dict]:
        if self._corpus is None:
            if self._invoice_id is not None:
                result = await self._db.execute(
                    text(
                        "SELECT id, chunk_text, page_num FROM invoice_chunks "
                        "WHERE invoice_id = :inv_id ORDER BY page_num"
                    ),
                    {"inv_id": str(self._invoice_id)},
                )
            else:
                result = await self._db.execute(
                    text(
                        "SELECT ic.id, ic.chunk_text, ic.page_num, ic.invoice_id, i.file_name "
                        "FROM invoice_chunks ic JOIN invoices i ON i.id = ic.invoice_id "
                        "WHERE ic.tenant_id = :tid AND i.file_type != 'image' "
                        "ORDER BY i.file_name, ic.page_num"
                    ),
                    {"tid": str(self._tenant_id)},
                )
            rows = await self._unwrap_rows(result)
            self._corpus = []
            for r in rows:
                entry = {"id": str(r["id"]), "chunk_text": r["chunk_text"], "page_num": r["page_num"]}
                if "file_name" in r.keys():
                    entry["invoice_id"] = str(r["invoice_id"])
                    entry["file_name"] = r["file_name"]
                self._corpus.append(entry)
        return self._corpus
```

`_bm25_retrieve` — pass metadata through (replace the returned dict comprehension):

```python
        return [
            {**corpus[i], "score": float(scores[i])}
            for i in ranked[:n]
        ]
```

(Note: corpus entries already use key `chunk_text`; keep that key here.)

`_dense_retrieve` — branch the SQL the same way:

```python
    async def _dense_retrieve(self, corpus: list[dict], query: str, n: int) -> list[dict]:
        query_embedding = await asyncio.to_thread(self._provider.embed_text, [query])
        query_embedding = query_embedding[0]
        if self._invoice_id is not None:
            sql = (
                "WITH q AS (SELECT CAST(:emb AS vector) AS vec) "
                "SELECT ic.id, ic.chunk_text, ic.page_num, "
                "1 - (ic.embedding <=> q.vec) AS similarity "
                "FROM invoice_chunks ic, q "
                "WHERE ic.invoice_id = :inv_id "
                "ORDER BY ic.embedding <=> q.vec LIMIT :n"
            )
            params = {"emb": str(query_embedding), "inv_id": str(self._invoice_id), "n": n}
        else:
            sql = (
                "WITH q AS (SELECT CAST(:emb AS vector) AS vec) "
                "SELECT ic.id, ic.chunk_text, ic.page_num, ic.invoice_id, i.file_name, "
                "1 - (ic.embedding <=> q.vec) AS similarity "
                "FROM invoice_chunks ic JOIN invoices i ON i.id = ic.invoice_id, q "
                "WHERE ic.tenant_id = :tid AND i.file_type != 'image' "
                "ORDER BY ic.embedding <=> q.vec LIMIT :n"
            )
            params = {"emb": str(query_embedding), "tid": str(self._tenant_id), "n": n}
        result = await self._db.execute(text(sql), params)
        rows = await self._unwrap_rows(result)
        out = []
        for r in rows:
            entry = {
                "id": str(r["id"]), "chunk_text": r["chunk_text"],
                "page_num": r["page_num"], "score": float(r["similarity"]),
            }
            if "file_name" in r.keys():
                entry["invoice_id"] = str(r["invoice_id"])
                entry["file_name"] = r["file_name"]
            out.append(entry)
        return out
```

`retrieve` — in the fused-list construction, propagate metadata:

```python
        fused = []
        for cid in all_ids:
            c = chunk_by_id[cid]
            entry = {
                "text": c["chunk_text"],
                "page": c["page_num"],
                "score": _rrf_score(
                    bm25_ranks.get(cid, sentinel),
                    dense_ranks.get(cid, sentinel),
                ),
            }
            if "file_name" in c:
                entry["invoice_id"] = c["invoice_id"]
                entry["file_name"] = c["file_name"]
            fused.append(entry)
```

Note: `FakeResult` rows are plain dicts — `r.keys()` works on both dicts and SQLAlchemy `RowMapping`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_retriever_tenant.py tests/unit/test_retriever.py -v`
Expected: all PASS (existing single-invoice tests must stay green)

- [ ] **Step 5: Commit**

```bash
git add agents/retriever.py tests/unit/test_retriever_tenant.py
git commit -m "feat(chat): tenant-wide hybrid retrieval with file_name metadata"
```

---

### Task 4: Chat agent graph

**Files:**
- Create: `agents/chat_agent.py`
- Test: `tests/unit/test_chat_agent.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_chat_agent.py`:

```python
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PDF_ID = str(uuid.uuid4())
IMG_ID = str(uuid.uuid4())
ROSTER = [
    {"id": PDF_ID, "file_name": "acme.pdf", "file_type": "pdf"},
    {"id": IMG_ID, "file_name": "receipt.jpg", "file_type": "image"},
]
CHUNK = {"text": "Invoice total: $500. Due 2026-08-01.", "page": 1, "score": 0.9,
         "invoice_id": PDF_ID, "file_name": "acme.pdf"}


def _agent(provider, retriever=None, roster=None, extractions=None, load_image=None):
    from agents.chat_agent import build_chat_agent
    retriever = retriever or AsyncMock(retrieve=AsyncMock(return_value=[CHUNK]))
    return build_chat_agent(
        retriever,
        provider,
        invoice_roster=roster or ROSTER,
        load_extractions=AsyncMock(return_value=extractions or []),
        load_image=load_image or AsyncMock(return_value=Path("fake.jpg")),
    )


def _invoke(agent, query, messages=None):
    from agents.chat_agent import make_initial_state
    return agent.ainvoke(make_initial_state(query, messages or []))


@pytest.mark.asyncio
async def test_condense_skipped_without_history():
    provider = MagicMock()
    provider.generate.side_effect = ["detail", "The total is $500 (acme.pdf, page 1).", "yes"]
    result = await _invoke(_agent(provider), "what is the total?")
    assert result["standalone_query"] == "what is the total?"
    assert provider.generate.call_count == 3  # route, answer, critique — no condense call


@pytest.mark.asyncio
async def test_condense_rewrites_with_history():
    provider = MagicMock()
    provider.generate.side_effect = [
        "What is the due date on the Acme invoice?",  # condense
        "detail",                                      # route
        "It is due 2026-08-01 (acme.pdf, page 1).",   # answer
        "yes",                                         # critique
    ]
    history = [
        {"role": "user", "content": "what is the total on acme.pdf?"},
        {"role": "assistant", "content": "The total is $500."},
    ]
    result = await _invoke(_agent(provider), "and when is it due?", history)
    assert result["standalone_query"] == "What is the due date on the Acme invoice?"
    assert "2026-08-01" in result["answer"]


@pytest.mark.asyncio
async def test_aggregate_route_answers_from_extraction_table():
    provider = MagicMock()
    provider.generate.side_effect = [
        "aggregate",
        "globex.pdf has the highest total: $900.",
    ]
    extractions = [
        {"file_name": "acme.pdf", "schema": {"vendor_name": "Acme", "invoice_number": "A-1",
                                             "invoice_date": "2026-06-01", "total_amount": 500, "currency": "USD"}},
        {"file_name": "globex.pdf", "schema": {"vendor_name": "Globex", "invoice_number": "G-9",
                                               "invoice_date": "2026-06-15", "total_amount": 900, "currency": "USD"}},
    ]
    result = await _invoke(_agent(provider, extractions=extractions), "which invoice has the highest total?")
    assert result["route"] == "aggregate"
    assert "globex" in result["answer"].lower()
    table_prompt = provider.generate.call_args_list[1][0][0]
    assert "acme.pdf" in table_prompt and "globex.pdf" in table_prompt


@pytest.mark.asyncio
async def test_aggregate_with_no_extractions_explains_without_llm_answer():
    provider = MagicMock()
    provider.generate.side_effect = ["aggregate"]
    extractions = [{"file_name": "acme.pdf", "schema": None}]
    result = await _invoke(_agent(provider, extractions=extractions), "what is my total spend?")
    assert "extract" in result["answer"].lower()
    assert provider.generate.call_count == 1  # routing only


@pytest.mark.asyncio
async def test_image_route_targets_named_image_invoice():
    provider = MagicMock()
    provider.generate.side_effect = ["image"]
    provider.generate_with_image = MagicMock(return_value="The receipt total is $12.50.")
    load_image = AsyncMock(return_value=Path("fake.jpg"))
    result = await _invoke(_agent(provider, load_image=load_image), "what is the total on receipt.jpg?")
    assert result["route"] == "image_detail"
    assert "12.50" in result["answer"]
    load_image.assert_awaited_once_with(IMG_ID)


@pytest.mark.asyncio
async def test_image_route_falls_back_to_detail_when_ambiguous():
    roster = [
        {"id": str(uuid.uuid4()), "file_name": "a.jpg", "file_type": "image"},
        {"id": str(uuid.uuid4()), "file_name": "b.jpg", "file_type": "image"},
    ]
    provider = MagicMock()
    provider.generate.side_effect = ["image", "I could not find that information in your invoices.", "yes"]
    retriever = AsyncMock(retrieve=AsyncMock(return_value=[]))
    result = await _invoke(_agent(provider, retriever=retriever, roster=roster), "what does the photo say?")
    assert result["route"] == "detail"


@pytest.mark.asyncio
async def test_critique_retries_ungrounded_answer():
    provider = MagicMock()
    provider.generate.side_effect = [
        "detail",
        "made-up answer",     # first answer
        "no",                 # critique rejects
        "The total is $500.", # retry answer
        "yes",                # critique accepts
    ]
    result = await _invoke(_agent(provider), "what is the total?")
    assert result["critique_iterations"] == 2
    assert result["grounded"] is True


@pytest.mark.asyncio
async def test_detail_answer_includes_sources():
    provider = MagicMock()
    provider.generate.side_effect = ["detail", "The total is $500 (acme.pdf, page 1).", "yes"]
    result = await _invoke(_agent(provider), "what is the total?")
    assert result["sources"] and result["sources"][0]["file_name"] == "acme.pdf"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_chat_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.chat_agent'`

- [ ] **Step 3: Implement `agents/chat_agent.py`**

```python
import asyncio
from pathlib import Path
from typing import Awaitable, Callable, TypedDict

from langgraph.graph import END, StateGraph

from agents.base import LLMProvider

HISTORY_WINDOW = 10
MAX_AGGREGATE_ROWS = 200


class ChatState(TypedDict):
    messages: list[dict]
    query: str
    standalone_query: str
    route: str
    target_invoice_id: str
    chunks: list[dict]
    answer: str
    grounded: bool
    critique_iterations: int
    sources: list[dict]


def make_initial_state(query: str, messages: list[dict]) -> ChatState:
    return {
        "messages": messages,
        "query": query,
        "standalone_query": "",
        "route": "",
        "target_invoice_id": "",
        "chunks": [],
        "answer": "",
        "grounded": False,
        "critique_iterations": 0,
        "sources": [],
    }


def _transcript(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages[-HISTORY_WINDOW:])


def build_chat_agent(
    retriever,
    provider: LLMProvider,
    *,
    invoice_roster: list[dict],
    load_extractions: Callable[[], Awaitable[list[dict]]],
    load_image: Callable[[str], Awaitable[Path]],
    max_critique: int = 2,
):
    async def condense_question(state: ChatState) -> ChatState:
        if not state["messages"]:
            return {**state, "standalone_query": state["query"]}
        prompt = (
            "Given the conversation below and a follow-up question, rewrite the "
            "follow-up as a standalone question that is fully understandable "
            "without the conversation. Return ONLY the rewritten question.\n\n"
            f"Conversation:\n{_transcript(state['messages'])}\n\n"
            f"Follow-up: {state['query']}\nStandalone question:"
        )
        standalone = provider.generate(prompt).strip()
        return {**state, "standalone_query": standalone or state["query"]}

    async def route_question(state: ChatState) -> ChatState:
        roster_lines = "\n".join(
            f"- {r['file_name']} (type: {r['file_type']})" for r in invoice_roster
        )
        prompt = (
            "You are routing a question about a set of invoices.\n"
            "Reply with EXACTLY one word:\n"
            "aggregate - the question compares invoices or asks about counts, sums, "
            "highest/lowest values, or which invoice has some property\n"
            "image - the question asks about the contents of one specific invoice "
            "whose type is 'image'\n"
            "detail - anything else\n\n"
            f"Invoices:\n{roster_lines}\n\n"
            f"Question: {state['standalone_query']}\nAnswer:"
        )
        verdict = provider.generate(prompt).strip().lower()
        if verdict.startswith("aggregate"):
            route = "aggregate"
        elif verdict.startswith("image"):
            route = "image_detail"
        else:
            route = "detail"

        target = ""
        if route == "image_detail":
            images = [r for r in invoice_roster if r["file_type"] == "image"]
            q = state["standalone_query"].lower()
            named = [r for r in images if Path(r["file_name"]).stem.lower() in q]
            if named:
                target = str(named[0]["id"])
            elif len(images) == 1:
                target = str(images[0]["id"])
            else:
                route = "detail"  # cannot identify which image — fall back to text RAG
        return {**state, "route": route, "target_invoice_id": target}

    async def aggregate_answer(state: ChatState) -> ChatState:
        rows = await load_extractions()
        extracted = [r for r in rows if r["schema"]]
        missing = [r["file_name"] for r in rows if not r["schema"]]
        if not extracted:
            return {
                **state,
                "answer": (
                    "None of your invoices have structured extractions yet. "
                    "Run extraction from the Extract tab first, then ask me again."
                ),
                "sources": [],
                "grounded": True,
            }
        lines = [
            "| File | Vendor | Invoice # | Date | Total | Currency |",
            "|---|---|---|---|---|---|",
        ]
        for r in extracted[:MAX_AGGREGATE_ROWS]:
            s = r["schema"]
            total = s.get("total_amount")
            lines.append(
                f"| {r['file_name']} | {s.get('vendor_name') or '—'} "
                f"| {s.get('invoice_number') or '—'} | {s.get('invoice_date') or '—'} "
                f"| {'—' if total is None else total} | {s.get('currency') or '—'} |"
            )
        table = "\n".join(lines)
        notes = ""
        if missing:
            notes += f"\nInvoices without extractions (not in the table): {', '.join(missing)}."
        if len(extracted) > MAX_AGGREGATE_ROWS:
            notes += f"\nOnly the first {MAX_AGGREGATE_ROWS} invoices are shown."
        prompt = (
            "Answer the question using ONLY the invoice table below. Be precise "
            "with numbers and name the invoices you refer to. If the table cannot "
            "answer the question, say so.\n\n"
            f"{table}{notes}\n\nQuestion: {state['standalone_query']}\nAnswer:"
        )
        answer = provider.generate(prompt).strip()
        sources = [
            {"file_name": r["file_name"], "page": None, "text": "structured extraction"}
            for r in extracted[:MAX_AGGREGATE_ROWS]
        ]
        return {**state, "answer": answer, "sources": sources, "grounded": True}

    async def retrieve(state: ChatState) -> ChatState:
        chunks = await retriever.retrieve(state["standalone_query"])
        return {**state, "chunks": chunks}

    async def generate_answer(state: ChatState) -> ChatState:
        if not state["chunks"]:
            return {
                **state,
                "answer": "I could not find anything relevant in your invoices.",
                "sources": [],
                "grounded": True,
            }
        context = "\n\n".join(
            f"[{c.get('file_name', 'invoice')} — page {c.get('page')}]\n{c['text']}"
            for c in state["chunks"]
        )
        prompt = (
            "Use the invoice excerpts below to answer the question. Cite the file "
            "name and page for every fact you state. If the answer is not in the "
            "excerpts, say 'I could not find that information in your invoices.'\n\n"
            f"Excerpts:\n{context}\n\nQuestion: {state['standalone_query']}\nAnswer:"
        )
        answer = provider.generate(prompt).strip()
        sources = [
            {"file_name": c.get("file_name"), "page": c.get("page"), "text": c["text"][:300]}
            for c in state["chunks"]
        ]
        return {**state, "answer": answer, "sources": sources}

    async def self_critique(state: ChatState) -> ChatState:
        context = "\n\n".join(c["text"] for c in state["chunks"])
        prompt = (
            f"Context:\n{context}\n\nAnswer: {state['answer']}\n\n"
            "Is this answer directly supported by the context? Reply ONLY 'yes' or 'no'."
        )
        verdict = provider.generate(prompt).strip().lower()
        return {
            **state,
            "grounded": verdict.startswith("yes"),
            "critique_iterations": state.get("critique_iterations", 0) + 1,
        }

    async def image_answer(state: ChatState) -> ChatState:
        image_path = await load_image(state["target_invoice_id"])
        prompt = ""
        if state["messages"]:
            prompt += f"Conversation so far:\n{_transcript(state['messages'])}\n\n"
        prompt += (
            "Answer the question using the attached invoice image. Be precise "
            f"with numbers.\n\nQuestion: {state['standalone_query']}\nAnswer:"
        )
        answer = await asyncio.to_thread(provider.generate_with_image, prompt, image_path)
        roster_by_id = {str(r["id"]): r["file_name"] for r in invoice_roster}
        file_name = roster_by_id.get(state["target_invoice_id"], "image invoice")
        return {
            **state,
            "answer": answer.strip(),
            "sources": [{"file_name": file_name, "page": None, "text": "vision analysis"}],
            "grounded": True,
        }

    def route_edge(state: ChatState) -> str:
        return state["route"]

    def answer_edge(state: ChatState) -> str:
        return "critique" if state["chunks"] else "end"

    def critique_edge(state: ChatState) -> str:
        if state["grounded"] or state.get("critique_iterations", 0) >= max_critique:
            return "end"
        return "retry"

    graph = StateGraph(ChatState)
    graph.add_node("condense_question", condense_question)
    graph.add_node("route_question", route_question)
    graph.add_node("aggregate_answer", aggregate_answer)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("self_critique", self_critique)
    graph.add_node("image_answer", image_answer)

    graph.set_entry_point("condense_question")
    graph.add_edge("condense_question", "route_question")
    graph.add_conditional_edges(
        "route_question", route_edge,
        {"aggregate": "aggregate_answer", "detail": "retrieve", "image_detail": "image_answer"},
    )
    graph.add_edge("aggregate_answer", END)
    graph.add_edge("retrieve", "generate_answer")
    graph.add_conditional_edges(
        "generate_answer", answer_edge,
        {"critique": "self_critique", "end": END},
    )
    graph.add_conditional_edges(
        "self_critique", critique_edge,
        {"end": END, "retry": "generate_answer"},
    )
    graph.add_edge("image_answer", END)
    return graph.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_chat_agent.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add agents/chat_agent.py tests/unit/test_chat_agent.py
git commit -m "feat(chat): routed conversational LangGraph agent (aggregate/detail/vision)"
```

---

### Task 5: Chat API schemas

**Files:**
- Create: `api/schemas/chat.py`
- Test: `tests/unit/test_chat_schemas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_chat_schemas.py`:

```python
import pytest
from pydantic import ValidationError


def test_message_in_requires_content():
    from api.schemas.chat import MessageIn
    with pytest.raises(ValidationError):
        MessageIn(content="")
    assert MessageIn(content="hello").content == "hello"


def test_conversation_create_title_optional():
    from api.schemas.chat import ConversationCreate
    assert ConversationCreate().title is None
    assert ConversationCreate(title="Invoices June").title == "Invoices June"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_chat_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `api/schemas/chat.py`**

```python
from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str | None = None


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_chat_schemas.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add api/schemas/chat.py tests/unit/test_chat_schemas.py
git commit -m "feat(chat): request schemas for chat endpoints"
```

---

### Task 6: Chat router + registration

**Files:**
- Create: `api/routers/chat.py`
- Modify: `api/main.py`
- Test: `tests/integration/test_api_chat.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_api_chat.py`:

```python
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

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


def _mock_db():
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


def _app_with_db(mock_db):
    from api.main import create_app
    from db.session import get_db

    app = create_app()

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    return app


def _exec_result(items):
    result = MagicMock()
    result.scalars.return_value.all.return_value = items
    return result


@pytest.mark.asyncio
async def test_create_conversation():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        mock_db = _mock_db()
        app = _app_with_db(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/chat/conversations",
                    json={"title": "June invoices"},
                    headers={"Authorization": "Bearer fake"},
                )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["title"] == "June invoices"
    assert mock_db.add.called and mock_db.commit.await_count == 1


@pytest.mark.asyncio
async def test_create_conversation_requires_auth():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        app = _app_with_db(_mock_db())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/chat/conversations", json={})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_conversations_empty():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        mock_db = _mock_db()
        mock_db.execute = AsyncMock(return_value=_exec_result([]))
        app = _app_with_db(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/chat/conversations", headers={"Authorization": "Bearer fake"}
                )
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_get_conversation_404():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        mock_db = _mock_db()
        mock_db.scalar = AsyncMock(return_value=None)
        app = _app_with_db(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/chat/conversations/{uuid.uuid4()}",
                    headers={"Authorization": "Bearer fake"},
                )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_send_message_persists_and_returns_answer():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        mock_db = _mock_db()
        conv = MagicMock(id=uuid.uuid4(), title="New conversation")
        mock_db.scalar = AsyncMock(return_value=conv)
        mock_db.execute = AsyncMock(return_value=_exec_result([]))
        app = _app_with_db(mock_db)
        fake_state = {"answer": "globex.pdf has the highest total: $900.",
                      "route": "aggregate", "sources": []}
        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            with patch("api.routers.chat._run_chat_agent", new=AsyncMock(return_value=fake_state)):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        f"/api/v1/chat/conversations/{conv.id}/messages",
                        json={"content": "which invoice has the highest total?"},
                        headers={"Authorization": "Bearer fake"},
                    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert "$900" in data["content"]
    assert data["meta"]["route"] == "aggregate"
    assert mock_db.add.call_count == 2  # user + assistant messages
    assert conv.title == "which invoice has the highest total?"[:80]


@pytest.mark.asyncio
async def test_send_message_agent_failure_returns_502():
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        mock_db = _mock_db()
        conv = MagicMock(id=uuid.uuid4(), title="New conversation")
        mock_db.scalar = AsyncMock(return_value=conv)
        mock_db.execute = AsyncMock(return_value=_exec_result([]))
        app = _app_with_db(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            with patch("api.routers.chat._run_chat_agent", new=AsyncMock(side_effect=RuntimeError("llm down"))):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        f"/api/v1/chat/conversations/{conv.id}/messages",
                        json={"content": "hi"},
                        headers={"Authorization": "Bearer fake"},
                    )
    assert response.status_code == 502
    assert mock_db.add.call_count == 0  # nothing persisted on failure


@pytest.mark.asyncio
async def test_api_user_role_forbidden():
    API_USER = {**FAKE_USER, "app_metadata": {**FAKE_USER["app_metadata"], "role": "api_user"}}
    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        app = _app_with_db(_mock_db())
        with patch("api.dependencies.verify_supabase_jwt", return_value=API_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/chat/conversations", json={}, headers={"Authorization": "Bearer fake"}
                )
    assert response.status_code == 403
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/integration/test_api_chat.py -v`
Expected: FAIL — 404s (router not registered) / `ModuleNotFoundError`

- [ ] **Step 3: Implement `api/routers/chat.py`**

```python
import asyncio
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base import get_provider
from agents.chat_agent import build_chat_agent, make_initial_state
from agents.retriever import HybridRetriever
from api.dependencies import CurrentUser, require_roles
from api.schemas.chat import ConversationCreate, MessageIn
from api.services.storage import download_file
from db.models import Conversation, ConversationMessage, Extraction, Invoice
from db.session import get_db

router = APIRouter(tags=["chat"])
log = structlog.get_logger()

_NO_INVOICES_ANSWER = (
    "You don't have any ingested invoices yet — upload one from the sidebar "
    "and I'll be able to answer questions about it."
)


def _conv_dict(conv: Conversation) -> dict:
    return {
        "id": str(conv.id),
        "title": conv.title,
        "created_at": str(conv.created_at),
        "updated_at": str(conv.updated_at),
    }


def _msg_dict(msg: ConversationMessage) -> dict:
    return {
        "id": str(msg.id),
        "role": msg.role,
        "content": msg.content,
        "meta": msg.meta,
        "created_at": str(msg.created_at),
    }


async def _get_conversation(db: AsyncSession, conversation_id: uuid.UUID, tenant_id: str) -> Conversation:
    conv = await db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == uuid.UUID(tenant_id),
        )
    )
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


async def _run_chat_agent(db: AsyncSession, tenant_id: str, history: list[dict], question: str) -> dict:
    """Run the chat agent; returns {'answer', 'route', 'sources'}."""
    tenant_uuid = uuid.UUID(tenant_id)
    invoices = (
        await db.execute(
            select(Invoice).where(Invoice.tenant_id == tenant_uuid, Invoice.status == "ready")
        )
    ).scalars().all()

    if not invoices:
        return {"answer": _NO_INVOICES_ANSWER, "route": "none", "sources": []}

    roster = [
        {"id": str(i.id), "file_name": i.file_name, "file_type": i.file_type}
        for i in invoices
    ]
    provider = get_provider()
    retriever = HybridRetriever(invoice_id=None, db=db, provider=provider, tenant_id=tenant_uuid)

    async def load_extractions() -> list[dict]:
        rows = (
            await db.execute(
                select(Invoice, Extraction)
                .outerjoin(Extraction, Extraction.invoice_id == Invoice.id)
                .where(Invoice.tenant_id == tenant_uuid, Invoice.status == "ready")
            )
        ).all()
        return [
            {"file_name": inv.file_name, "schema": ext.schema_json if ext else None}
            for inv, ext in rows
        ]

    async def load_image(invoice_id: str) -> Path:
        inv = next(i for i in invoices if str(i.id) == invoice_id)
        content = await asyncio.to_thread(download_file, str(inv.tenant_id), inv.storage_path)
        suffix = "." + inv.file_name.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            return Path(tmp.name)

    agent = build_chat_agent(
        retriever, provider,
        invoice_roster=roster,
        load_extractions=load_extractions,
        load_image=load_image,
    )
    final = await agent.ainvoke(make_initial_state(question, history))
    return {
        "answer": final.get("answer", "No answer generated."),
        "route": final.get("route", ""),
        "sources": final.get("sources", []),
    }


@router.post("/chat/conversations", response_model=dict)
async def create_conversation(
    body: ConversationCreate,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    user_uuid = None
    try:
        user_uuid = uuid.UUID(user.id)
    except (ValueError, AttributeError):
        pass
    conv = Conversation(
        id=uuid.uuid4(),
        tenant_id=uuid.UUID(user.tenant_id),
        user_id=user_uuid,
        title=body.title or "New conversation",
        created_at=now,
        updated_at=now,
    )
    db.add(conv)
    await db.commit()
    return {"data": _conv_dict(conv), "error": None, "request_id": None}


@router.get("/chat/conversations", response_model=dict)
async def list_conversations(
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    convs = (
        await db.execute(
            select(Conversation)
            .where(Conversation.tenant_id == uuid.UUID(user.tenant_id))
            .order_by(Conversation.updated_at.desc())
        )
    ).scalars().all()
    return {"data": [_conv_dict(c) for c in convs], "error": None, "request_id": None}


@router.get("/chat/conversations/{conversation_id}", response_model=dict)
async def get_conversation(
    conversation_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    conv = await _get_conversation(db, conversation_id, user.tenant_id)
    messages = (
        await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conv.id)
            .order_by(ConversationMessage.created_at)
        )
    ).scalars().all()
    return {
        "data": {**_conv_dict(conv), "messages": [_msg_dict(m) for m in messages]},
        "error": None,
        "request_id": None,
    }


@router.delete("/chat/conversations/{conversation_id}", response_model=dict)
async def delete_conversation(
    conversation_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    conv = await _get_conversation(db, conversation_id, user.tenant_id)
    await db.delete(conv)
    await db.commit()
    return {"data": {"deleted": str(conversation_id)}, "error": None, "request_id": None}


@router.post("/chat/conversations/{conversation_id}/messages", response_model=dict)
async def send_message(
    conversation_id: uuid.UUID,
    body: MessageIn,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    conv = await _get_conversation(db, conversation_id, user.tenant_id)
    tenant_uuid = uuid.UUID(user.tenant_id)

    history_rows = (
        await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conv.id)
            .order_by(ConversationMessage.created_at)
        )
    ).scalars().all()
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    try:
        result = await _run_chat_agent(db, user.tenant_id, history, body.content)
    except Exception as exc:
        log.error("chat.agent_failed", conversation_id=str(conversation_id), error=str(exc))
        raise HTTPException(502, f"Chat agent failed: {exc}")

    now = datetime.now(timezone.utc)
    user_msg = ConversationMessage(
        id=uuid.uuid4(), conversation_id=conv.id, tenant_id=tenant_uuid,
        role="user", content=body.content, created_at=now,
    )
    asst_msg = ConversationMessage(
        id=uuid.uuid4(), conversation_id=conv.id, tenant_id=tenant_uuid,
        role="assistant", content=result["answer"],
        meta={"route": result["route"], "sources": result["sources"]},
        created_at=now,
    )
    db.add(user_msg)
    db.add(asst_msg)
    if conv.title == "New conversation":
        conv.title = body.content[:80]
    conv.updated_at = now
    await db.commit()
    return {"data": _msg_dict(asst_msg), "error": None, "request_id": None}
```

- [ ] **Step 4: Register the router in `api/main.py`**

Add to the imports block:

```python
from api.routers import chat as chat_router
```

Add after the audit router include (keep the same prefix pattern):

```python
    app.include_router(chat_router.router, prefix="/api/v1")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/integration/test_api_chat.py -v`
Expected: 7 PASS

- [ ] **Step 6: Run the full integration suite for regressions**

Run: `python -m pytest tests/integration/ -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add api/routers/chat.py api/main.py tests/integration/test_api_chat.py
git commit -m "feat(chat): conversation CRUD and message endpoints with persisted turns"
```

---

### Task 7: API client chat methods

**Files:**
- Modify: `frontend/api_client.py`
- Test: `tests/unit/test_api_client_chat.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_api_client_chat.py`:

```python
from unittest.mock import AsyncMock

import pytest

from frontend.api_client import APIClient


@pytest.mark.asyncio
async def test_chat_client_methods_hit_expected_paths():
    client = APIClient("http://x", "tok")
    client._post = AsyncMock(return_value={"id": "c1"})
    client._get = AsyncMock(return_value=[])
    client._delete = AsyncMock(return_value={"deleted": "c1"})

    await client.create_conversation("June")
    client._post.assert_awaited_with("/api/v1/chat/conversations", json={"title": "June"})

    await client.list_conversations()
    client._get.assert_awaited_with("/api/v1/chat/conversations")

    await client.get_conversation("c1")
    client._get.assert_awaited_with("/api/v1/chat/conversations/c1")

    await client.send_message("c1", "highest total?")
    client._post.assert_awaited_with(
        "/api/v1/chat/conversations/c1/messages", json={"content": "highest total?"}
    )

    await client.delete_conversation("c1")
    client._delete.assert_awaited_with("/api/v1/chat/conversations/c1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_api_client_chat.py -v`
Expected: FAIL — `AttributeError: 'APIClient' object has no attribute 'create_conversation'`

- [ ] **Step 3: Implement client methods**

In `frontend/api_client.py`, add a `_delete` helper after `_post`:

```python
    async def _delete(self, path: str) -> Any:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(f"{self._base}{path}", headers=self._headers())
            if not resp.is_success:
                raise Exception(f"{resp.status_code}: {resp.text}")
            return resp.json()["data"]
```

Add at the end of the class:

```python
    async def create_conversation(self, title: str | None = None) -> dict:
        return await self._post("/api/v1/chat/conversations", json={"title": title})

    async def list_conversations(self) -> list:
        return await self._get("/api/v1/chat/conversations")

    async def get_conversation(self, conversation_id: str) -> dict:
        return await self._get(f"/api/v1/chat/conversations/{conversation_id}")

    async def send_message(self, conversation_id: str, content: str) -> dict:
        return await self._post(
            f"/api/v1/chat/conversations/{conversation_id}/messages", json={"content": content}
        )

    async def delete_conversation(self, conversation_id: str) -> dict:
        return await self._delete(f"/api/v1/chat/conversations/{conversation_id}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_api_client_chat.py tests/unit/test_api_client.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/api_client.py tests/unit/test_api_client_chat.py
git commit -m "feat(chat): API client methods for conversations"
```

---

### Task 8: Liquid glass theme

**Files:**
- Create: `frontend/theme.py`
- Modify: `frontend/app.py` (inject only — Chat tab comes in Task 9)
- Test: `tests/unit/test_theme.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_theme.py`:

```python
def test_theme_exports_inject():
    from frontend.theme import inject_theme, _THEME_CSS
    assert callable(inject_theme)


def test_theme_css_has_glass_and_motion_safety():
    from frontend.theme import _THEME_CSS
    assert "backdrop-filter" in _THEME_CSS
    assert "prefers-reduced-motion" in _THEME_CSS
    assert "typing-dots" in _THEME_CSS
    assert "@keyframes" in _THEME_CSS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_theme.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'frontend.theme'`

- [ ] **Step 3: Implement `frontend/theme.py`**

```python
import streamlit as st

_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Outfit:wght@400;500;600&display=swap');

:root {
    --glass-bg: rgba(44, 43, 40, 0.55);
    --glass-bg-strong: rgba(38, 37, 34, 0.72);
    --glass-border: rgba(255, 255, 255, 0.09);
    --glass-highlight: rgba(255, 255, 255, 0.22);
    --accent: #D97757;
    --ink: #ECEAE4;
    --ink-dim: #A8A599;
}

/* Typography */
html, body,
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]),
[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]) {
    font-family: 'Outfit', 'Segoe UI', sans-serif;
    color: var(--ink);
}
h1, h2, h3, [data-testid="stMetricValue"] {
    font-family: 'Fraunces', Georgia, serif !important;
    letter-spacing: -0.015em;
}

/* Animated depth backdrop — gives the glass something to refract */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(60% 80% at 15% 10%, rgba(217, 119, 87, 0.14), transparent 60%),
        radial-gradient(50% 70% at 85% 85%, rgba(120, 140, 200, 0.10), transparent 60%),
        radial-gradient(40% 55% at 70% 20%, rgba(217, 170, 87, 0.07), transparent 55%),
        #1D1C19;
    background-size: 160% 160%;
}

/* Glass surfaces */
[data-testid="stSidebar"],
[data-testid="stChatMessage"],
[data-testid="stMetric"],
[data-testid="stExpander"] details,
.glass-panel {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid var(--glass-border);
    border-top-color: var(--glass-highlight);
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
[data-testid="stSidebar"] {
    border-radius: 0;
    border-top-color: var(--glass-border);
    border-right: 1px solid var(--glass-border);
}
[data-testid="stChatMessage"] {
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.35rem;
}

/* Chat input pinned bar */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background: var(--glass-bg-strong) !important;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border-radius: 16px;
    border: 1px solid var(--glass-border);
}

/* Buttons — lift, sheen, spring */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    font-weight: 600;
    letter-spacing: 0.01em;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--glass-border);
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(217, 119, 87, 0.28);
    border-color: rgba(217, 119, 87, 0.45);
}
.stButton > button:active, .stDownloadButton > button:active {
    transform: scale(0.97);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.02em;
    font-size: 0.95rem;
}

/* Metric numbers */
[data-testid="stMetric"] { padding: 0.9rem 1.1rem; }
[data-testid="stMetricValue"], [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }

/* Typing indicator */
.typing-dots { display: inline-flex; gap: 6px; padding: 6px 2px; }
.typing-dots span {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--ink-dim);
    opacity: 0.3;
}

/* Empty state panel */
.glass-empty {
    background: var(--glass-bg);
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid var(--glass-border);
    border-top-color: var(--glass-highlight);
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-top: 0.8rem;
}
.glass-empty h3 { margin: 0 0 0.5rem 0; }
.glass-empty p { color: var(--ink-dim); margin: 0.2rem 0; max-width: 38rem; }

/* Motion — only for users who haven't opted out */
@media (prefers-reduced-motion: no-preference) {
    [data-testid="stAppViewContainer"] {
        animation: glassDrift 26s ease-in-out infinite alternate;
    }
    [data-testid="stChatMessage"] {
        animation: msgIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .stButton > button, .stDownloadButton > button {
        transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1),
                    box-shadow 0.2s ease, border-color 0.2s ease;
    }
    button[data-baseweb="tab"] { transition: color 0.18s ease; }
    .typing-dots span { animation: dotPulse 1.2s ease-in-out infinite; }
    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
}

@keyframes glassDrift {
    from { background-position: 0% 0%; }
    to { background-position: 100% 100%; }
}
@keyframes msgIn {
    from { opacity: 0; transform: translateY(14px) scale(0.98); }
    to { opacity: 1; transform: none; }
}
@keyframes dotPulse {
    0%, 60%, 100% { opacity: 0.25; transform: translateY(0); }
    30% { opacity: 1; transform: translateY(-4px); }
}
</style>
"""


def inject_theme() -> None:
    """Apply the liquid-glass theme. Call once per page render, right after set_page_config."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
```

- [ ] **Step 4: Inject in `frontend/app.py`**

After `st.set_page_config(...)` add:

```python
from frontend.theme import inject_theme
inject_theme()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_theme.py -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/theme.py frontend/app.py tests/unit/test_theme.py
git commit -m "feat(frontend): liquid glass theme with motion-safe animations"
```

---

### Task 9: Chat page UI

**Files:**
- Create: `frontend/pages/chat.py`
- Modify: `frontend/app.py`
- Test: `tests/unit/test_chat_page.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_chat_page.py`:

```python
def test_chat_page_exports_render():
    from frontend.pages.chat import render
    assert callable(render)


def test_typing_indicator_markup():
    from frontend.pages.chat import _TYPING_HTML
    assert "typing-dots" in _TYPING_HTML
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_chat_page.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `frontend/pages/chat.py`**

```python
import asyncio

import streamlit as st

from frontend.api_client import APIClient

_TYPING_HTML = '<div class="typing-dots"><span></span><span></span><span></span></div>'

_EMPTY_HTML = """
<div class="glass-empty">
  <h3>Ask across every invoice</h3>
  <p>Start a new conversation and ask things like
  <em>"Which invoice has the highest total?"</em> or
  <em>"When is the Acme invoice due?"</em> — follow-up questions welcome.</p>
</div>
"""


def _run(coro):
    return asyncio.run(coro)


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if msg["role"] == "assistant" and meta.get("sources"):
            st.caption(f"route: {meta.get('route', '—')}")
            with st.expander("Sources"):
                for s in meta["sources"]:
                    page = f" — page {s['page']}" if s.get("page") else ""
                    st.markdown(f"**{s.get('file_name') or 'invoice'}**{page}")
                    if s.get("text"):
                        st.text(s["text"][:300])


def render(client: APIClient):
    st.subheader("Chat")
    try:
        conversations = _run(client.list_conversations())
    except Exception as e:
        st.error(str(e))
        return

    picker_col, thread_col = st.columns([1, 3], gap="large")

    with picker_col:
        if st.button("New conversation", type="primary", use_container_width=True):
            conv = _run(client.create_conversation())
            st.session_state["chat_conversation_id"] = conv["id"]
            st.rerun()
        for conv in conversations:
            active = st.session_state.get("chat_conversation_id") == conv["id"]
            title_col, del_col = st.columns([5, 1])
            label = (conv.get("title") or "Untitled")[:40]
            if title_col.button(("● " if active else "") + label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state["chat_conversation_id"] = conv["id"]
                st.rerun()
            if del_col.button("✕", key=f"delconv_{conv['id']}", help="Delete conversation"):
                _run(client.delete_conversation(conv["id"]))
                if active:
                    st.session_state.pop("chat_conversation_id", None)
                st.rerun()

    conv_id = st.session_state.get("chat_conversation_id")

    with thread_col:
        if not conv_id:
            st.markdown(_EMPTY_HTML, unsafe_allow_html=True)
        else:
            try:
                detail = _run(client.get_conversation(conv_id))
            except Exception as e:
                st.error(str(e))
                st.session_state.pop("chat_conversation_id", None)
                return
            for msg in detail.get("messages", []):
                _render_message(msg)

    if conv_id:
        question = st.chat_input("Ask about your invoices…")
        if question:
            with thread_col:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown(_TYPING_HTML, unsafe_allow_html=True)
                    try:
                        reply = _run(client.send_message(conv_id, question))
                        placeholder.markdown(reply["content"])
                    except Exception as e:
                        placeholder.error(str(e))
                        return
            st.rerun()
```

- [ ] **Step 4: Add the Chat tab in `frontend/app.py`**

Replace the tabs block:

```python
chat_tab, qa_tab, extract_tab, compare_tab, batch_tab, dashboard_tab = st.tabs(
    ["Chat", "Q&A", "Extract", "Compare", "Batch", "Dashboard"]
)

from frontend.pages import chat, qa, extract, compare, batch, dashboard
with chat_tab:
    chat.render(client)
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

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_chat_page.py -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/pages/chat.py frontend/app.py tests/unit/test_chat_page.py
git commit -m "feat(frontend): conversational chat page with liquid glass styling"
```

---

### Task 10: README + full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the feature**

In `README.md`, add these bullets at the top of the `## Features` list:

```markdown
- **Conversational Cross-Invoice Chat (API v2)** — persisted multi-turn conversations over all ingested invoices; a routed LangGraph agent condenses follow-up questions against history, answers aggregate questions ("which invoice has the highest total?") from structured extractions, cites file name + page for document details, and sends image invoices to Gemini vision
- **Liquid Glass UI** — the API-backed Streamlit frontend ships a glassmorphic theme: blurred translucent panels, animated depth backdrop, sprung message entrances, and a typing indicator (motion-safe via `prefers-reduced-motion`)
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests/ -m "not slow" -v`
Expected: all PASS (existing + new). If anything fails, fix before committing.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document conversational chat and liquid glass theme"
```

---

## Verification checklist (post-implementation)

- [ ] `python -m pytest tests/ -m "not slow"` — green
- [ ] `python -m alembic history` shows `0001 -> 0002 (head)`
- [ ] `python -c "from api.main import create_app"` imports cleanly (with env vars set)
- [ ] Manual (optional, needs stack): create conversation → ask aggregate question → ask follow-up → verify condensed routing in `meta`
