import asyncio
import re
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
            named = [
                r for r in images
                if re.search(rf"\b{re.escape(Path(r['file_name']).stem.lower())}\b", q)
                or r["file_name"].lower() in q
            ]
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
