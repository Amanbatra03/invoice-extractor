import asyncio
import time
import streamlit as st
from frontend.api_client import APIClient

_STATUS_MAP = {
    "ready":     "badge-ready",
    "pending":   "badge-pending",
    "ingesting": "badge-pending",
    "failed":    "badge-failed",
}


def _badge(status: str) -> str:
    cls = _STATUS_MAP.get(status.lower(), "badge")
    return f'<span class="badge {cls}">{status.upper()}</span>'


def render(client: APIClient):
    st.markdown('<div class="brut-header">BATCH EXTRACT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brut-sub">SELECT MULTIPLE INVOICES · EXTRACT ALL FIELDS AT ONCE</div>',
        unsafe_allow_html=True,
    )

    try:
        invoices_data = asyncio.run(client.list_invoices(limit=100))
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return

    if not invoices:
        st.markdown(
            '<div style="border:2px solid #2A2A2A;padding:1.5rem;color:#666;font-size:0.78rem;">'
            'NO INVOICES FOUND. UPLOAD FROM THE INVOICES PAGE.</div>',
            unsafe_allow_html=True,
        )
        return

    # SELECT ALL / DESELECT ALL
    sel_col, desel_col, _ = st.columns([1, 1, 5])
    with sel_col:
        if st.button("SELECT ALL", use_container_width=True):
            for inv in invoices:
                st.session_state[f"batch_{inv['id']}"] = True
            st.rerun()
    with desel_col:
        if st.button("DESELECT ALL", use_container_width=True):
            for inv in invoices:
                st.session_state[f"batch_{inv['id']}"] = False
            st.rerun()

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    selected = []
    for inv in invoices:
        inv_status = inv.get("status", "")
        label = inv["file_name"]
        checked = st.checkbox(label, key=f"batch_{inv['id']}")
        if checked:
            selected.append(inv["id"])

    st.markdown(
        f'<div style="color:#F5F500;font-size:0.82rem;font-weight:700;'
        f'letter-spacing:0.06em;margin:0.75rem 0;">{len(selected)} SELECTED</div>',
        unsafe_allow_html=True,
    )

    btn_col, _ = st.columns([2, 5])
    with btn_col:
        if st.button(
            "RUN BATCH EXTRACTION",
            type="primary",
            disabled=len(selected) == 0,
            use_container_width=True,
        ):
            with st.spinner("QUEUING BATCH JOB..."):
                try:
                    result = asyncio.run(client.batch_extract(selected))
                    batch_job_id = result["batch_job_id"]
                except Exception as e:
                    st.error(str(e))
                    return

            st.markdown(
                f'<div style="border:2px solid #4488FF;padding:0.75rem 1rem;'
                f'font-size:0.78rem;color:#4488FF;margin-bottom:0.5rem;">'
                f'BATCH JOB QUEUED: {batch_job_id}</div>',
                unsafe_allow_html=True,
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            for iteration in range(60):
                time.sleep(3)
                try:
                    job = asyncio.run(client.get_job(batch_job_id))
                except Exception as e:
                    st.error(f"POLLING FAILED: {e}")
                    break

                progress_bar.progress(min(0.95, iteration / 60))
                status_text.markdown(
                    f'<div style="color:#666;font-size:0.75rem;letter-spacing:0.06em;">'
                    f'STATUS: {job["status"].upper()}...</div>',
                    unsafe_allow_html=True,
                )

                if job["status"] in ("done", "failed"):
                    progress_bar.progress(1.0)
                    status_text.empty()
                    if job["status"] == "done":
                        r = job.get("result", {})
                        st.markdown(
                            f'<div style="border:2px solid #00FF88;padding:1rem;'
                            f'color:#00FF88;font-size:0.82rem;font-weight:700;">'
                            f'✓ DONE — {r.get("success_count", 0)} SUCCEEDED, '
                            f'{r.get("failure_count", 0)} FAILED</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(f"BATCH FAILED: {job.get('error', 'UNKNOWN ERROR')}")
                    break
            else:
                status_text.markdown(
                    '<div style="color:#FF8C00;font-size:0.78rem;">POLLING TIMEOUT — CHECK JOBS ON INVOICES PAGE.</div>',
                    unsafe_allow_html=True,
                )
