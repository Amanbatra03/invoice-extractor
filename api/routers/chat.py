import asyncio
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
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


@router.post("/chat/conversations/{conversation_id}/messages/stream")
async def send_message_stream(
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

    now = datetime.now(timezone.utc)
    user_msg = ConversationMessage(
        id=uuid.uuid4(), conversation_id=conv.id, tenant_id=tenant_uuid,
        role="user", content=body.content, created_at=now,
    )
    db.add(user_msg)
    await db.commit()

    async def event_generator() -> AsyncGenerator[str, None]:
        final_state: dict = {}
        answer = _NO_INVOICES_ANSWER
        try:
            invoices = (
                await db.execute(
                    select(Invoice).where(Invoice.tenant_id == tenant_uuid, Invoice.status == "ready")
                )
            ).scalars().all()

            if not invoices:
                yield f"data: {json.dumps({'node': 'answer', 'answer': _NO_INVOICES_ANSWER, 'route': 'none', 'sources': []})}\n\n"
            else:
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
                async for event in agent.astream(make_initial_state(body.content, history)):
                    node_name = list(event.keys())[0]
                    state = list(event.values())[0]
                    final_state = state
                    yield f"data: {json.dumps({'node': node_name, 'answer': state.get('answer', ''), 'route': state.get('route', ''), 'sources': state.get('sources', [])})}\n\n"
                answer = final_state.get("answer", "No answer generated.")

            asst_msg = ConversationMessage(
                id=uuid.uuid4(), conversation_id=conv.id, tenant_id=tenant_uuid,
                role="assistant", content=answer,
                meta={"route": final_state.get("route", ""), "sources": final_state.get("sources", [])},
                created_at=datetime.now(timezone.utc),
            )
            db.add(asst_msg)
            if conv.title == "New conversation":
                conv.title = body.content[:80]
            conv.updated_at = datetime.now(timezone.utc)
            await db.commit()
            yield f"data: {json.dumps({'node': '__done__', 'message_id': str(asst_msg.id)})}\n\n"

        except Exception as exc:
            log.error("chat.stream_failed", conversation_id=str(conversation_id), error=str(exc))
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
