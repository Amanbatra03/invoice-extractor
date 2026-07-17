import asyncio
import time
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.markdown(
        "### Batch Extract"
        "<br><span style='color:#A8A599;font-size:0.85rem;'>"
        "Select multiple invoices and extract all fields in one job.</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    try:
        invoices_data = asyncio.run(client.list_invoices(limit=100))
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return
    if not invoices:
        st.info("No invoices loaded yet.")
        return
    selected = [inv["id"] for inv in invoices if st.checkbox(inv["file_name"], key=f"batch_{inv['id']}")]
    st.caption(f"{len(selected)} invoice(s) selected")
    if st.button("Run Batch Extraction", type="primary", disabled=len(selected) == 0):
        with st.spinner("Queuing batch job..."):
            try:
                result = asyncio.run(client.batch_extract(selected))
                batch_job_id = result["batch_job_id"]
                st.info(f"Batch job queued: `{batch_job_id}`")
                progress = st.progress(0)
                status_text = st.empty()
                for _ in range(60):
                    time.sleep(3)
                    job = asyncio.run(client.get_job(batch_job_id))
                    if job["status"] in ("done", "failed"):
                        progress.progress(1.0)
                        if job["status"] == "done":
                            r = job.get("result", {})
                            st.success(f"Done — {r.get('success_count', 0)} succeeded, {r.get('failure_count', 0)} failed")
                        else:
                            st.error(f"Batch failed: {job.get('error')}")
                        break
                    status_text.text(f"Status: {job['status']}...")
            except Exception as e:
                st.error(str(e))
