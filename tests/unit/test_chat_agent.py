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
