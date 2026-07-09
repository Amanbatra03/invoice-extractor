# Conversational Cross-Invoice Chat — Design

**Date:** 2026-07-10
**Status:** Approved
**Target:** Enterprise layer (FastAPI `api/` + `agents/` + Streamlit `frontend/`)

---

## 1. Problem Statement

The current Q&A is single-turn and single-invoice: `POST /invoices/{id}/qa` carries only the current question, the agent state has no history, and the frontend replaces the previous answer on every ask. Users cannot ask follow-up questions ("and when is *it* due?") or cross-invoice questions ("which of my invoices has the highest total?"). Image invoices are ingested as a single placeholder chunk (`workers/ingest_job.py`), so text retrieval over them is empty.

This feature adds persisted, multi-turn, cross-invoice conversational chat with a routed agent, plus a premium "liquid glass" visual refresh of the Streamlit frontend.

## 2. Requirements

### Functional
- Multi-turn chat: follow-up questions are resolved against conversation history.
- Cross-invoice scope: one conversation spans all of the tenant's ingested invoices.
- Aggregate questions ("highest total", "same vendor", "total spend") answered from structured `extractions`, not chunk retrieval.
- Document-detail questions answered via cross-invoice hybrid retrieval with per-invoice citations (file name + page).
- Detail questions about image invoices answered via a vision call (image + history).
- Conversations persisted per tenant/user; list, resume, delete.
- Existing `/qa` endpoints remain unchanged (backward compatible).

### Non-Functional
- Tenant isolation on every new table and query (matches existing conventions).
- RBAC: `admin`, `analyst`, `viewer` can chat (same as `/qa`).
- Works with both providers (`GeminiProvider`, `OllamaGemmaProvider`).
- Frontend: Apple liquid-glass aesthetic with fluid animations; respects `prefers-reduced-motion`.
- TDD: unit tests per agent node + retriever mode; integration tests per endpoint.

## 3. Database (Alembic migration 0002)

Two tables in `db/models.py`, following existing column conventions:

```
conversations
  id UUID PK · tenant_id UUID FK tenants (CASCADE, indexed)
  user_id UUID FK users (SET NULL) · title VARCHAR(255)  -- first user message, truncated
  created_at / updated_at TIMESTAMPTZ

conversation_messages
  id UUID PK · conversation_id UUID FK conversations (CASCADE, indexed)
  tenant_id UUID (indexed) · role VARCHAR(10)  -- 'user' | 'assistant'
  content TEXT · meta JSONB NULL  -- sources, route taken, trace
  created_at TIMESTAMPTZ
```

## 4. Retriever: tenant-wide mode

`agents/retriever.py` — `HybridRetriever.__init__` gains `tenant_id: uuid.UUID` and makes `invoice_id` optional:

- `invoice_id` set → current behavior, unchanged.
- `invoice_id=None` → corpus/pgvector queries filter `WHERE tenant_id = :tid`, joined to `invoices` for `file_name`; excludes image placeholder chunks (`file_type != 'image'` via join). Result dicts gain `invoice_id` and `file_name`.

## 5. Chat agent (`agents/chat_agent.py`)

New LangGraph graph; `agents/qa_agent.py` is untouched.

```
START → condense_question → route_question ─┬→ aggregate_answer ────────────┐
                                            ├→ retrieve → generate_answer   │
                                            │       → self_critique ──────→ END
                                            └→ image_answer ────────────────┘
```

State: `messages` (history window), `query`, `standalone_query`, `route`, `chunks`, `answer`, `grounded`, `critique_iterations`, `sources`.

- **condense_question** — last 10 messages + new question → standalone query. If there is no history, pass the question through unchanged (no LLM call).
- **route_question** — LLM classification into `aggregate` | `detail` | `image_detail`. `image_detail` is chosen when the standalone query names a specific invoice whose `file_type == "image"` (invoice roster with file types is provided in the routing prompt). Unparseable verdicts default to `detail`.
- **aggregate_answer** — loads all tenant extractions (`schema_json`), builds a compact markdown table (file name, vendor, invoice #, date, total, currency), answers over it. Invoices without extractions are listed as "not yet extracted" so the model does not hallucinate them. Guard: if the table exceeds ~200 rows, truncate and say so in the prompt.
- **retrieve** — tenant-wide hybrid retrieval (Section 4).
- **generate_answer** — grounded answer; prompt requires citing file name and page for each claim.
- **self_critique** — reuse the existing yes/no grounding check with `MAX_CRITIQUE_ITERATIONS`.
- **image_answer** — downloads the image via `api/services/storage.download_file`, calls `provider.generate_with_image(prompt, image_path)` with condensed question + history.

### Provider protocol addition

`agents/base.LLMProvider` gains:

```python
def generate_with_image(self, prompt: str, image_path: Path, system: str | None = None) -> str: ...
```

- `GeminiProvider`: native multimodal `generate_content([prompt, PIL.Image])`.
- `OllamaGemmaProvider`: Ollama `images=[...]` parameter (Gemma 3 is multimodal); if the local model rejects images, return a clear "image questions require the Gemini provider" message rather than raising.

## 6. API (`api/routers/chat.py`)

Same conventions as existing routers: envelope `{data, error, request_id}`, `require_roles("admin", "analyst", "viewer")`, tenant scoping on every query, audit middleware captures writes, rate limits via the shared limiter.

| Method | Path | Behavior |
|---|---|---|
| POST | `/api/v1/chat/conversations` | Create; optional `title` |
| GET | `/api/v1/chat/conversations` | List (tenant + user scoped, newest first) |
| GET | `/api/v1/chat/conversations/{id}` | Conversation + messages |
| POST | `/api/v1/chat/conversations/{id}/messages` | Body `{content}`. Loads history, runs agent, persists user + assistant messages (assistant `meta` = sources/route/trace), bumps `conversations.updated_at`, returns assistant message |
| DELETE | `/api/v1/chat/conversations/{id}` | Delete (cascade) |

Schemas in `api/schemas/chat.py` (`ConversationOut`, `MessageIn`, `MessageOut`). Router registered in `api/main.py`.

## 7. Frontend — chat page + liquid glass theme

### Chat page (`frontend/pages/chat.py`)
- Conversation picker (new conversation / resume existing / delete) backed by the new endpoints.
- `st.chat_message` for the thread, `st.chat_input` for entry.
- While the agent runs: animated typing indicator (three pulsing dots) in the assistant bubble.
- Under each assistant reply: "Sources" expander (file name, page, snippet) and "Route" caption (aggregate / detail / image).
- New `APIClient` methods: `create_conversation`, `list_conversations`, `get_conversation`, `send_message`, `delete_conversation`.

### Liquid glass theme (`frontend/theme.py`)
Single module exposing `inject_theme()`, called from `frontend/app.py` so every page inherits it:

- **Glass surfaces**: sidebar, chat bubbles, cards, expanders — translucent layered backgrounds (`rgba` fills), `backdrop-filter: blur(24px) saturate(180%)`, 1px inner specular border (light top-left, dark bottom-right), soft ambient shadow.
- **Depth**: animated slow-drifting gradient backdrop behind the glass so blur has something to refract.
- **Motion**: chat messages enter with a spring-like translate/fade keyframe; buttons and list rows have hover lift + specular sweep; typing indicator with staggered dot pulse. All transitions 150–350 ms, eased (`cubic-bezier`), and wrapped in `@media (prefers-reduced-motion: no-preference)`.
- **Typography/color**: keep the existing Fraunces/Outfit pairing and the warm dark palette from `app.py` so both frontends stay visually related; glass tints derive from it.
- Implementation is pure CSS injected via `st.markdown` — no JS, so it degrades gracefully.

## 8. Error handling

- No invoices ingested → assistant answers: upload invoices first (no LLM call).
- Aggregate route with zero extractions → explains extraction hasn't been run, offers the Extract page.
- Provider/agent exception → 502 envelope error, matching other routers; message is **not** persisted so the user can retry cleanly.
- Conversation not found / cross-tenant access → 404.
- History window capped at 10 messages; conversation content itself is unlimited.

## 9. Testing

- **Unit** (`tests/unit/test_chat_agent.py`, extend `test_retriever.py`, `test_schemas.py`): condense pass-through vs rewrite, router verdict parsing + image routing + default-to-detail, aggregate table building (missing extractions, truncation), citation prompt assembly, retriever tenant-mode SQL filters and metadata, provider `generate_with_image` fallbacks. Fake provider + fake DB rows, mirroring existing tests.
- **Integration** (`tests/integration/test_api_chat.py`): CRUD lifecycle, message round-trip persists two rows with meta, tenant isolation (404 across tenants), RBAC (`api_user` role gets 403, matching `/qa`), error envelope on provider failure.
- Frontend theme is CSS-only; verified visually via `streamlit run`.

## 10. Out of scope

- Streaming chat responses (existing `/qa/stream` pattern can be added later).
- Conversation sharing between users, titles via LLM summarization.
- Tool-calling/SQL-agent approaches.
- Reworking image ingestion (placeholder chunk stays; vision path compensates).
