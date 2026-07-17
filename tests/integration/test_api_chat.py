import json
import os
import uuid
from datetime import datetime, timezone
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


# ── streaming SSE endpoint ───────────────────────────────────────────────────

def _fake_invoice(tenant_id: str):
    inv = MagicMock()
    inv.id = uuid.uuid4()
    inv.tenant_id = uuid.UUID(tenant_id)
    inv.file_name = "inv.pdf"
    inv.file_type = "pdf"
    inv.storage_path = "path/inv.pdf"
    inv.status = "ready"
    return inv


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


@pytest.mark.asyncio
async def test_send_message_stream_yields_sse_events_and_persists():
    tenant_id = FAKE_USER["app_metadata"]["tenant_id"]
    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.tenant_id = uuid.UUID(tenant_id)
    conv.title = "New conversation"
    conv.updated_at = datetime.now(timezone.utc)

    mock_db = _mock_db()
    mock_db.scalar = AsyncMock(return_value=conv)
    inv = _fake_invoice(tenant_id)
    mock_db.execute = AsyncMock(side_effect=[
        _exec_result([]),        # history query
        _exec_result([inv]),     # invoices query inside generator
    ])

    # async generator that simulates two LangGraph node events
    async def _fake_astream(initial_state):
        yield {"condense_question": {**initial_state, "standalone_query": "total?"}}
        yield {"generate_answer": {
            **initial_state,
            "standalone_query": "total?",
            "answer": "Total is $500.",
            "sources": [{"file_name": "inv.pdf", "page": 1, "text": "..."}],
            "grounded": True,
        }}

    mock_agent = MagicMock()
    mock_agent.astream = _fake_astream

    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        app = _app_with_db(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER), \
             patch("api.routers.chat.build_chat_agent", return_value=mock_agent), \
             patch("api.routers.chat.get_provider", return_value=MagicMock()), \
             patch("api.routers.chat.HybridRetriever", return_value=MagicMock()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    f"/api/v1/chat/conversations/{conv_id}/messages/stream",
                    json={"content": "what is the total?"},
                    headers={"Authorization": "Bearer fake"},
                )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse(response.text)
    assert any(e.get("node") == "generate_answer" for e in events), events
    assert any(e.get("answer") == "Total is $500." for e in events), events

    done_events = [e for e in events if e.get("node") == "__done__"]
    assert len(done_events) == 1
    assert "message_id" in done_events[0]

    # user msg + assistant msg persisted
    assert mock_db.add.call_count == 2
    assert mock_db.commit.await_count == 2
    assert conv.title == "what is the total?"[:80]


@pytest.mark.asyncio
async def test_send_message_stream_no_invoices_yields_fallback_and_persists():
    tenant_id = FAKE_USER["app_metadata"]["tenant_id"]
    conv_id = uuid.uuid4()
    conv = MagicMock()
    conv.id = conv_id
    conv.tenant_id = uuid.UUID(tenant_id)
    conv.title = "My chat"
    conv.updated_at = datetime.now(timezone.utc)

    mock_db = _mock_db()
    mock_db.scalar = AsyncMock(return_value=conv)
    mock_db.execute = AsyncMock(side_effect=[
        _exec_result([]),    # history query
        _exec_result([]),    # invoices query — empty
    ])

    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        app = _app_with_db(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    f"/api/v1/chat/conversations/{conv_id}/messages/stream",
                    json={"content": "summarise my invoices"},
                    headers={"Authorization": "Bearer fake"},
                )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    answer_events = [e for e in events if e.get("node") == "answer"]
    assert len(answer_events) == 1
    assert "upload" in answer_events[0]["answer"].lower() or "don't have" in answer_events[0]["answer"].lower()
    # still persists user + assistant
    assert mock_db.add.call_count == 2


@pytest.mark.asyncio
async def test_send_message_stream_conv_not_found_returns_404():
    mock_db = _mock_db()
    mock_db.scalar = AsyncMock(return_value=None)

    with patch.dict(os.environ, _TEST_ENV):
        from api.config import get_settings
        get_settings.cache_clear()
        app = _app_with_db(mock_db)
        with patch("api.dependencies.verify_supabase_jwt", return_value=FAKE_USER):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    f"/api/v1/chat/conversations/{uuid.uuid4()}/messages/stream",
                    json={"content": "hi"},
                    headers={"Authorization": "Bearer fake"},
                )

    assert response.status_code == 404
