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
