import asyncio
import time
import streamlit as st
from frontend.api_client import APIClient

_STATUS_MAP = {
    "ready":     "badge-ready",
    "pending":   "badge-pending",
    "ingesting": "badge-pending",
    "failed":    "badge-failed",
    "running":   "badge-running",
}

_MAX_POLLS = 20  # 20 × 3s = 60s max wait


def _badge(status: str) -> str:
    cls = _STATUS_MAP.get(status.lower(), "badge")
    return f'<span class="badge {cls}">{status.upper()}</span>'


def _clear_poll_state() -> None:
    for k in ("batch_polling", "batch_job_id", "batch_poll_count", "batch_poll_progress"):
        st.session_state.pop(k, None)


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

    # ── Polling state: check job status once per rerun, then rerun again ────
    if st.session_state.get("batch_polling"):
        batch_job_id = st.session_state["batch_job_id"]
        poll_count = st.session_state.get("batch_poll_count", 0)
        progress_val = st.session_state.get("batch_poll_progress", 0.05)

        st.markdown(
            f'<div style="border:2px solid #4488FF;padding:0.75rem 1rem;'
            f'font-size:0.78rem;color:#4488FF;margin-bottom:1rem;">'
            f'BATCH JOB RUNNING: {batch_job_id}</div>',
            unsafe_allow_html=True,
        )
        progress_bar = st.progress(progress_val)
        status_text = st.empty()

        try:
            job = asyncio.run(client.get_job(batch_job_id))
            job_status = job.get("status", "")
        except Exception as e:
            st.error(f"POLLING FAILED: {e}")
            _clear_poll_state()
            return

        poll_count += 1
        st.session_state["batch_poll_count"] = poll_count
        st.session_state["batch_poll_progress"] = min(0.95, poll_count / _MAX_POLLS)

        status_text.markdown(
            f'<div style="color:#666;font-size:0.75rem;letter-spacing:0.06em;">'
            f'STATUS: {job_status.upper()} — CHECK {poll_count}/{_MAX_POLLS}</div>',
            unsafe_allow_html=True,
        )

        if job_status in ("done", "failed"):
            progress_bar.progress(1.0)
            status_text.empty()
            _clear_poll_state()
            if job_status == "done":
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
            if st.button("START NEW BATCH", key="batch_reset"):
                st.rerun()
        elif poll_count >= _MAX_POLLS:
            progress_bar.progress(1.0)
            status_text.markdown(
                '<div style="color:#FF8C00;font-size:0.78rem;">'
                'POLLING TIMEOUT — CHECK JOBS ON INVOICES PAGE.</div>',
                unsafe_allow_html=True,
            )
            _clear_poll_state()
        else:
            # Still running — sleep once (3s) then let Streamlit rerun
            time.sleep(3)
            st.rerun()
        return

    # ── Selection UI ──────────────────────────────────────────────────────────
    ready_invoices = [inv for inv in invoices if inv.get("status") == "ready"]

    sel_col, desel_col, _ = st.columns([1, 1, 5])
    with sel_col:
        if st.button("SELECT ALL", use_container_width=True, key="batch_sel_all"):
            for inv in ready_invoices:
                st.session_state[f"batch_{inv.get('id', '')}"] = True
            st.rerun()
    with desel_col:
        if st.button("DESELECT ALL", use_container_width=True, key="batch_desel_all"):
            for inv in invoices:
                st.session_state[f"batch_{inv.get('id', '')}"] = False
            st.rerun()

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    selected = []
    for inv in invoices:
        inv_id = inv.get("id", "")
        inv_status = inv.get("status", "")
        is_ready = inv_status == "ready"
        file_name = inv.get("file_name", "")
        label = f"{file_name}  [{inv_status.upper()}]"
        checked = st.checkbox(
            label,
            key=f"batch_{inv_id}",
            disabled=not is_ready,
        )
        if checked and is_ready:
            selected.append(inv_id)

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
            key="batch_run",
        ):
            with st.spinner("QUEUING BATCH JOB..."):
                try:
                    result = asyncio.run(client.batch_extract(selected))
                    batch_job_id = result.get("batch_job_id")
                    if not batch_job_id:
                        st.error("NO JOB ID RETURNED FROM API.")
                        return
                    st.session_state["batch_job_id"] = batch_job_id
                    st.session_state["batch_polling"] = True
                    st.session_state["batch_poll_count"] = 0
                    st.session_state["batch_poll_progress"] = 0.05
                except Exception as e:
                    st.error(str(e))
                    return
            st.rerun()
