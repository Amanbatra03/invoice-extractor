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
            "agent_trace": [
                {k: str(v)[:300] for k, v in list(step.values())[0].items()}
                for step in trace_steps
            ],
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
        import json
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
            node_name = list(event.keys())[0]
            state = list(event.values())[0]
            yield f"data: {json.dumps({'node': node_name, 'answer': state.get('answer', '')})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
