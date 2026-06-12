from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.base import LLMProvider
from agents.retriever import HybridRetriever


class QAState(TypedDict):
    query: str
    rewritten_query: str
    chunks: list[dict]
    answer: str
    relevant: bool
    grounded: bool
    iterations: int
    critique_iterations: int


def build_qa_agent(
    retriever: HybridRetriever,
    provider: LLMProvider,
    max_iter: int | None = None,
    max_critique: int | None = None,
):
    if max_iter is None or max_critique is None:
        try:
            from api.config import get_settings
            settings = get_settings()
            max_iter = max_iter or settings.MAX_AGENT_ITERATIONS
            max_critique = max_critique or settings.MAX_CRITIQUE_ITERATIONS
        except Exception:
            max_iter = max_iter or 3
            max_critique = max_critique or 2

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
