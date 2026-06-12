from typing import TypedDict

from langgraph.graph import StateGraph, END

from rag.hybrid_retriever import HybridRetriever
from rag.utils import load_config


class AgentState(TypedDict):
    query: str
    rewritten_query: str
    chunks: list[dict]
    answer: str
    relevant: bool
    grounded: bool
    iterations: int
    critique_iterations: int


def build_agent(retriever: HybridRetriever, llm=None):
    cfg = load_config()

    if llm is None:
        from rag.llm import get_ollama_llm
        llm = get_ollama_llm(cfg.LLM, num_ctx=int(cfg.NUM_CTX))

    def query_rewriter(state: AgentState) -> AgentState:
        prompt = (
            f"Rewrite this invoice question to be specific and extractable.\n"
            f"Original: {state['query']}\n"
        )
        if state.get("rewritten_query"):
            prompt += (
                f"A previous rewrite '{state['rewritten_query']}' retrieved irrelevant "
                f"context; produce a substantively different phrasing.\n"
            )
        prompt += "Rewritten:"
        rewritten = llm.invoke(prompt).strip()
        return {
            **state,
            "rewritten_query": rewritten,
            "iterations": state.get("iterations", 0) + 1,
        }

    def hybrid_retrieve(state: AgentState) -> AgentState:
        chunks = retriever.retrieve(state["rewritten_query"])
        return {**state, "chunks": chunks}

    def relevance_grade(state: AgentState) -> AgentState:
        if not state["chunks"]:
            return {**state, "relevant": False}
        context = "\n".join(c["text"][:200] for c in state["chunks"])
        prompt = (
            f"Query: {state['rewritten_query']}\n"
            f"Context: {context}\n"
            f"Is the context relevant to answer the query? Reply ONLY 'yes' or 'no'."
        )
        verdict = llm.invoke(prompt).strip().lower()
        return {**state, "relevant": verdict.startswith("yes")}

    def route_from_grade(state: AgentState) -> str:
        if state["relevant"]:
            return "generate"
        if state.get("iterations", 0) >= cfg.MAX_AGENT_ITERATIONS:
            return "generate"
        return "retry"

    def generate_answer(state: AgentState) -> AgentState:
        context = "\n\n".join(c["text"] for c in state["chunks"])
        prompt = (
            f"Use the following invoice context to answer the question.\n"
            f"If the answer is not present, say 'I could not find that information in the invoice.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['query']}\nAnswer:"
        )
        answer = llm.invoke(prompt).strip()
        return {**state, "answer": answer}

    def self_critique(state: AgentState) -> AgentState:
        context = "\n\n".join(c["text"] for c in state["chunks"])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer: {state['answer']}\n\n"
            f"Is this answer directly supported by the context? Reply ONLY 'yes' or 'no'."
        )
        verdict = llm.invoke(prompt).strip().lower()
        return {
            **state,
            "grounded": verdict.startswith("yes"),
            "critique_iterations": state.get("critique_iterations", 0) + 1,
        }

    def route_from_critique(state: AgentState) -> str:
        if state["grounded"]:
            return "end"
        if state.get("critique_iterations", 0) > cfg.MAX_CRITIQUE_ITERATIONS:
            return "end"
        return "retry"

    graph = StateGraph(AgentState)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("relevance_grade", relevance_grade)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("self_critique", self_critique)

    graph.set_entry_point("query_rewriter")
    graph.add_edge("query_rewriter", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "relevance_grade")
    graph.add_conditional_edges(
        "relevance_grade",
        route_from_grade,
        {"generate": "generate_answer", "retry": "query_rewriter"},
    )
    graph.add_edge("generate_answer", "self_critique")
    graph.add_conditional_edges(
        "self_critique",
        route_from_critique,
        {"end": END, "retry": "generate_answer"},
    )

    return graph.compile()
