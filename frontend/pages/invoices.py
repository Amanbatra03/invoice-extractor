import asyncio
import streamlit as st
from frontend.api_client import APIClient

_STATUS_MAP = {
    "ready":     "badge-ready",
    "pending":   "badge-pending",
    "ingesting": "badge-pending",
    "failed":    "badge-failed",
    "running":   "badge-running",
}


def _badge(status: str) -> str:
    cls = _STATUS_MAP.get(status.lower(), "badge")
    return f'<span class="badge {cls}">{status.upper()}</span>'


def _date(iso: str) -> str:
    return iso[:10] if iso else "—"


def render(client: APIClient):
    hdr_col, refresh_col = st.columns([8, 1])
    with hdr_col:
        st.markdown('<div class="brut-header">INVOICES</div>', unsafe_allow_html=True)
    with refresh_col:
        st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
        if st.button("↺ REFRESH", use_container_width=True):
            st.rerun()

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown('<div class="brut-sub">UPLOAD DOCUMENT</div>', unsafe_allow_html=True)
    upload_col, btn_col = st.columns([4, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Select file",
            type=["pdf", "jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    with btn_col:
        st.markdown('<div style="margin-top:1.6rem;"></div>', unsafe_allow_html=True)
        if st.button("UPLOAD", type="primary", disabled=uploaded is None, use_container_width=True):
            try:
                content = uploaded.read()
                result = asyncio.run(
                    client.upload_invoice(
                        uploaded.name,
                        content,
                        uploaded.type or "application/octet-stream",
                    )
                )
                job_id = result.get("job_id") or result.get("invoice_id", "—")
                st.success(f"QUEUED — JOB {job_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"UPLOAD FAILED: {exc}")

    st.divider()

    # ── Load invoices ───────────────────────────────────────────────────────
    try:
        data = asyncio.run(client.list_invoices(page=1, limit=200))
        items = data.get("items", [])
        total = data.get("total", 0)
    except Exception as exc:
        st.error(f"FAILED TO LOAD INVOICES: {exc}")
        items, total = [], 0

    ready_count = sum(1 for i in items if i.get("status") == "ready")
    processing = sum(1 for i in items if i.get("status") in ("pending", "ingesting"))
    failed = sum(1 for i in items if i.get("status") == "failed")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TOTAL", total)
    m2.metric("READY", ready_count)
    m3.metric("PROCESSING", processing)
    m4.metric("FAILED", failed)

    st.divider()

    # ── Invoice list ────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.62rem;letter-spacing:0.12em;color:#444;margin-bottom:0.5rem;'
        'font-family:\'JetBrains Mono\',monospace;">'
        'NAME · STATUS · DATE · ACTIONS</div>',
        unsafe_allow_html=True,
    )

    if not items:
        st.markdown(
            '<div style="border:2px solid #2A2A2A;padding:2rem;text-align:center;">'
            '<div style="color:#444;font-size:0.82rem;letter-spacing:0.06em;">'
            'NO INVOICES FOUND. UPLOAD A FILE ABOVE.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        for inv in items:
            card_col, qa_col, ext_col, del_col = st.columns([12, 1, 1, 1])

            with card_col:
                ftype = inv.get("file_type", "").upper()
                st.markdown(
                    f'<div class="invoice-card">'
                    f'{_badge(inv.get("status", ""))}'
                    f'<span style="flex:1;font-size:0.82rem;overflow:hidden;white-space:nowrap;'
                    f'text-overflow:ellipsis;">{inv.get("file_name", "")}</span>'
                    f'<span style="color:#444;font-size:0.68rem;margin-right:0.5rem;">[{ftype}]</span>'
                    f'<span style="color:#444;font-size:0.72rem;">{_date(inv.get("created_at",""))}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            inv_id = inv["id"]
            inv_status = inv.get("status", "")
            is_ready = inv_status == "ready"

            with qa_col:
                if st.button(
                    "Q&A",
                    key=f"qa_{inv_id}",
                    help="Ask a question about this invoice",
                    disabled=not is_ready,
                    use_container_width=True,
                ):
                    st.session_state["preselect_invoice_id"] = inv_id
                    st.session_state["nav"] = "qa"
                    st.rerun()

            with ext_col:
                can_extract = inv_status in ("ready", "failed")
                if st.button(
                    "EXT",
                    key=f"ext_{inv_id}",
                    help="Extract structured fields",
                    disabled=not can_extract,
                    use_container_width=True,
                ):
                    st.session_state["preselect_invoice_id"] = inv_id
                    st.session_state["nav"] = "extract"
                    st.rerun()

            with del_col:
                if st.button("✕", key=f"del_{inv_id}", help="Delete invoice", use_container_width=True):
                    try:
                        asyncio.run(client.delete_invoice(inv_id))
                        st.rerun()
                    except Exception as exc:
                        st.error(f"DELETE FAILED: {exc}")

    st.divider()

    # ── Recent jobs ─────────────────────────────────────────────────────────
    with st.expander("RECENT JOBS"):
        try:
            jobs = asyncio.run(client.list_jobs(limit=50))
            if not jobs:
                st.markdown(
                    '<div style="color:#444;font-size:0.78rem;">NO JOBS YET.</div>',
                    unsafe_allow_html=True,
                )
            else:
                for job in jobs[:15]:
                    st.markdown(
                        f'<div style="display:flex;gap:1rem;align-items:center;'
                        f'padding:0.35rem 0;border-bottom:1px solid #1E1E1E;">'
                        f'<span style="font-size:0.78rem;min-width:80px;">{job.get("type","").upper()}</span>'
                        f'{_badge(job.get("status",""))}'
                        f'<span style="color:#444;font-size:0.7rem;margin-left:auto;">'
                        f'{_date(job.get("created_at",""))}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        except Exception as exc:
            st.error(f"FAILED TO LOAD JOBS: {exc}")
