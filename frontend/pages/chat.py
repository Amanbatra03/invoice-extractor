import asyncio
import streamlit as st
from frontend.api_client import APIClient

_TYPING_HTML = '<div class="typing-dots"><span></span><span></span><span></span></div>'


def _run(coro):
    return asyncio.run(coro)


def _role_label(role: str) -> str:
    color = "#F5F500" if role == "user" else "#4488FF"
    text = "YOU" if role == "user" else "AI"
    return (
        f'<div style="font-size:0.62rem;letter-spacing:0.12em;color:{color};'
        f'font-weight:700;margin-bottom:0.25rem;font-family:\'JetBrains Mono\',monospace;">'
        f'{text}</div>'
    )


def _render_message(msg: dict) -> None:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(_role_label(role), unsafe_allow_html=True)
        st.markdown(msg["content"])
        meta = msg.get("meta") or {}
        if role == "assistant" and meta.get("sources"):
            with st.expander("SOURCES"):
                for s in meta["sources"]:
                    page = f" — PAGE {s['page']}" if s.get("page") else ""
                    st.markdown(
                        f'<span style="color:#F5F500;font-size:0.78rem;">'
                        f'{(s.get("file_name") or "INVOICE").upper()}{page}</span>',
                        unsafe_allow_html=True,
                    )
                    if s.get("text"):
                        st.code(s["text"][:300], language=None)


def render(client: APIClient):
    st.markdown('<div class="brut-header">CHAT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brut-sub">MULTI-TURN CONVERSATION · ALL INVOICES</div>',
        unsafe_allow_html=True,
    )

    try:
        conversations = _run(client.list_conversations())
    except Exception as e:
        st.error(str(e))
        return

    picker_col, thread_col = st.columns([1, 3])

    with picker_col:
        if st.button("+ NEW", type="primary", use_container_width=True):
            try:
                conv = _run(client.create_conversation())
                st.session_state["chat_conversation_id"] = conv["id"]
                st.rerun()
            except Exception as e:
                st.error(str(e))

        st.markdown(
            '<div style="font-size:0.62rem;letter-spacing:0.1em;color:#444;'
            'padding:0.6rem 0 0.3rem;font-family:\'JetBrains Mono\',monospace;">CONVERSATIONS</div>',
            unsafe_allow_html=True,
        )

        if not conversations:
            st.markdown(
                '<div style="color:#444;font-size:0.72rem;padding:0.5rem 0;'
                'font-family:\'JetBrains Mono\',monospace;">NONE YET</div>',
                unsafe_allow_html=True,
            )
        else:
            for conv in conversations:
                active = st.session_state.get("chat_conversation_id") == conv["id"]
                label = (conv.get("title") or "UNTITLED")[:32].upper()
                title_col, del_col = st.columns([5, 1])
                prefix = "▶ " if active else "   "
                if title_col.button(prefix + label, key=f"conv_{conv['id']}", use_container_width=True):
                    st.session_state["chat_conversation_id"] = conv["id"]
                    st.rerun()
                if del_col.button("✕", key=f"delconv_{conv['id']}", help="Delete"):
                    try:
                        _run(client.delete_conversation(conv["id"]))
                        if active:
                            st.session_state.pop("chat_conversation_id", None)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    conv_id = st.session_state.get("chat_conversation_id")

    with thread_col:
        if not conv_id:
            st.markdown(
                '<div style="border:2px solid #2A2A2A;padding:2.5rem;text-align:center;margin-top:1rem;">'
                '<div style="color:#F5F500;font-size:0.72rem;letter-spacing:0.1em;font-family:\'JetBrains Mono\',monospace;">SELECT OR CREATE A CONVERSATION</div>'
                '<div style="color:#444;font-size:0.72rem;margin-top:0.5rem;font-family:\'JetBrains Mono\',monospace;">'
                'ASK ACROSS ALL YOUR INVOICES — FOLLOW-UPS WELCOME</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                detail = _run(client.get_conversation(conv_id))
            except Exception as e:
                st.error(f"COULD NOT LOAD CONVERSATION — IT MAY HAVE BEEN DELETED. ({e})")
                st.session_state.pop("chat_conversation_id", None)
                return
            for msg in detail.get("messages", []):
                _render_message(msg)

    if conv_id:
        question = st.chat_input("ASK ABOUT YOUR INVOICES...")
        if question:
            with thread_col:
                with st.chat_message("user"):
                    st.markdown(_role_label("user"), unsafe_allow_html=True)
                    st.markdown(question)
                with st.chat_message("assistant"):
                    placeholder_label = st.empty()
                    placeholder_content = st.empty()
                    placeholder_label.markdown(_role_label("assistant"), unsafe_allow_html=True)
                    placeholder_content.markdown(_TYPING_HTML, unsafe_allow_html=True)
                    try:
                        reply = _run(client.send_message(conv_id, question))
                        placeholder_content.markdown(reply["content"])
                        meta = reply.get("meta") or {}
                        if meta.get("sources"):
                            with st.expander("SOURCES"):
                                for s in meta["sources"]:
                                    page = f" — PAGE {s['page']}" if s.get("page") else ""
                                    st.markdown(
                                        f'<span style="color:#F5F500;font-size:0.78rem;">'
                                        f'{(s.get("file_name") or "INVOICE").upper()}{page}</span>',
                                        unsafe_allow_html=True,
                                    )
                                    if s.get("text"):
                                        st.code(s["text"][:300], language=None)
                    except Exception as e:
                        placeholder_content.error(str(e))
                        return
            st.rerun()
