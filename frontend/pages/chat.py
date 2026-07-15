import asyncio

import streamlit as st

from frontend.api_client import APIClient

_TYPING_HTML = '<div class="typing-dots"><span></span><span></span><span></span></div>'

_EMPTY_HTML = """
<div class="glass-empty">
  <h3>Ask across every invoice</h3>
  <p>Start a new conversation and ask things like
  <em>"Which invoice has the highest total?"</em> or
  <em>"When is the Acme invoice due?"</em> — follow-up questions welcome.</p>
</div>
"""


def _run(coro):
    return asyncio.run(coro)


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if msg["role"] == "assistant" and meta.get("sources"):
            st.caption(f"route: {meta.get('route', '—')}")
            with st.expander("Sources"):
                for s in meta["sources"]:
                    page = f" — page {s['page']}" if s.get("page") else ""
                    st.markdown(f"**{s.get('file_name') or 'invoice'}**{page}")
                    if s.get("text"):
                        st.text(s["text"][:300])


def render(client: APIClient):
    st.subheader("Chat")
    try:
        conversations = _run(client.list_conversations())
    except Exception as e:
        st.error(str(e))
        return

    picker_col, thread_col = st.columns([1, 3], gap="large")

    with picker_col:
        if st.button("New conversation", type="primary", use_container_width=True):
            conv = _run(client.create_conversation())
            st.session_state["chat_conversation_id"] = conv["id"]
            st.rerun()
        for conv in conversations:
            active = st.session_state.get("chat_conversation_id") == conv["id"]
            title_col, del_col = st.columns([5, 1])
            label = (conv.get("title") or "Untitled")[:40]
            if title_col.button(("● " if active else "") + label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state["chat_conversation_id"] = conv["id"]
                st.rerun()
            if del_col.button("✕", key=f"delconv_{conv['id']}", help="Delete conversation"):
                _run(client.delete_conversation(conv["id"]))
                if active:
                    st.session_state.pop("chat_conversation_id", None)
                st.rerun()

    conv_id = st.session_state.get("chat_conversation_id")

    with thread_col:
        if not conv_id:
            st.markdown(_EMPTY_HTML, unsafe_allow_html=True)
        else:
            try:
                detail = _run(client.get_conversation(conv_id))
            except Exception as e:
                st.error(str(e))
                st.session_state.pop("chat_conversation_id", None)
                return
            for msg in detail.get("messages", []):
                _render_message(msg)

    if conv_id:
        question = st.chat_input("Ask about your invoices…")
        if question:
            with thread_col:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown(_TYPING_HTML, unsafe_allow_html=True)
                    try:
                        reply = _run(client.send_message(conv_id, question))
                        placeholder.markdown(reply["content"])
                    except Exception as e:
                        placeholder.error(str(e))
                        return
            st.rerun()
