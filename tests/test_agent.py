import pytest
from unittest.mock import MagicMock
from rag.agent import build_agent, AgentState


class _MockLLM:
    def __init__(self, responses: dict):
        self._responses = responses

    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for keyword, response in self._responses.items():
            if keyword in prompt_lower:
                return response
        return "default response"


def _make_mock_retriever(chunks):
    r = MagicMock()
    r.retrieve.return_value = chunks
    return r


def test_agent_happy_path():
    chunks = [{"text": "Total amount is $110.00", "page": 1, "score": 0.9}]
    # Keywords match actual prompt text: "rewrite", "relevant", "use the following", "supported"
    llm = _MockLLM({
        "rewrite": "What is the total amount?",
        "relevant": "yes",
        "use the following": "The total is $110.00",
        "supported": "yes",
    })
    retriever = _make_mock_retriever(chunks)
    agent = build_agent(retriever, llm=llm)

    result = agent.invoke({
        "query": "What is the total?",
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    })

    assert "answer" in result
    assert result["answer"] != ""


def test_agent_retries_on_irrelevant_then_gives_up():
    llm = _MockLLM({
        "rewrite": "What is the invoice number?",
        "relevant": "no",
        "use the following": "I could not find that information.",
        "supported": "yes",
    })
    retriever = _make_mock_retriever([{"text": "unrelated text", "page": 1, "score": 0.1}])
    agent = build_agent(retriever, llm=llm)

    result = agent.invoke({
        "query": "What is the invoice number?",
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    })

    assert result["iterations"] >= 1
    assert "answer" in result


def test_agent_self_critique_accepts_grounded_answer():
    chunks = [{"text": "Invoice date: 2024-01-15", "page": 1, "score": 0.95}]
    llm = _MockLLM({
        "rewrite": "What is the invoice date?",
        "relevant": "yes",
        "use the following": "The invoice date is 2024-01-15.",
        "supported": "yes",
    })
    retriever = _make_mock_retriever(chunks)
    agent = build_agent(retriever, llm=llm)

    result = agent.invoke({
        "query": "What is the invoice date?",
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    })

    assert result["grounded"] is True


class _RecordingLLM:
    def __init__(self, responses: dict):
        self._responses = responses
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        prompt_lower = prompt.lower()
        for keyword, response in self._responses.items():
            if keyword in prompt_lower:
                return response
        return "default response"


def test_rewrite_retry_includes_previous_attempt():
    llm = _RecordingLLM({
        "rewrite": "What is the invoice number?",
        "relevant": "no",
        "use the following": "not found",
        "supported": "yes",
    })
    retriever = _make_mock_retriever([{"text": "unrelated", "page": 1, "score": 0.1}])
    agent = build_agent(retriever, llm=llm)
    agent.invoke({
        "query": "invoice number?", "rewritten_query": "", "chunks": [],
        "answer": "", "relevant": False, "grounded": False,
        "iterations": 0, "critique_iterations": 0,
    })
    rewrite_prompts = [p for p in llm.prompts if "rewrite" in p.lower()]
    assert len(rewrite_prompts) >= 2
    assert "previous rewrite" in rewrite_prompts[1].lower()
