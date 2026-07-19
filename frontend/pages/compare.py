import asyncio
import pandas as pd
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.markdown('<div class="brut-header">COMPARE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brut-sub">SIDE-BY-SIDE DIFF · DISCREPANCIES HIGHLIGHTED IN RED</div>',
        unsafe_allow_html=True,
    )

    try:
        invoices_data = asyncio.run(client.list_invoices(limit=100))
        # Show only invoices that are ready (have been ingested + could have extractions)
        invoices = [i for i in invoices_data.get("items", []) if i.get("status") == "ready"]
    except Exception as e:
        st.error(str(e))
        return

    if len(invoices) < 2:
        st.markdown(
            '<div style="border:2px solid #2A2A2A;padding:1.5rem;color:#666;font-size:0.78rem;">'
            'NEED AT LEAST 2 READY INVOICES WITH EXTRACTIONS. UPLOAD AND EXTRACT FROM THE INVOICES PAGE.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div style="font-size:0.72rem;color:#666;letter-spacing:0.06em;margin-bottom:0.75rem;">'
        'SELECT 2 OR MORE INVOICES TO COMPARE (MUST HAVE EXTRACTIONS RUN FIRST)</div>',
        unsafe_allow_html=True,
    )

    selected = []
    for inv in invoices:
        file_type = inv.get("file_type", "").upper()
        label = f"{inv['file_name']}  [{file_type}]"
        if st.checkbox(label, key=f"cmp_{inv['id']}"):
            selected.append(inv["id"])

    count_col, btn_col, _ = st.columns([1, 2, 5])
    with count_col:
        st.markdown(
            f'<div style="color:#F5F500;font-size:0.82rem;font-weight:700;'
            f'padding-top:0.5rem;">{len(selected)} SELECTED</div>',
            unsafe_allow_html=True,
        )
    with btn_col:
        if st.button("RUN COMPARE", type="primary", disabled=len(selected) < 2, use_container_width=True):
            with st.spinner("COMPARING..."):
                try:
                    result = asyncio.run(client.compare_invoices(selected))
                    table = result.get("table", {})
                    discrepancies = result.get("discrepancies", [])

                    if table:
                        disc_fields = {d["field"] for d in discrepancies}
                        rows = [{"FIELD": f, **v} for f, v in table.items()]
                        df = pd.DataFrame(rows).set_index("FIELD")

                        def highlight(frame):
                            styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
                            for f in disc_fields:
                                if f in styles.index:
                                    styles.loc[f] = "background-color:#2A0000;color:#FF3333;"
                            return styles

                        st.markdown(
                            '<div class="brut-sub" style="margin-top:1.5rem;">COMPARISON TABLE</div>',
                            unsafe_allow_html=True,
                        )
                        st.dataframe(df.style.apply(highlight, axis=None), use_container_width=True)

                    if discrepancies:
                        st.markdown(
                            f'<div class="brut-sub" style="margin-top:1.5rem;color:#FF3333;">'
                            f'{len(discrepancies)} DISCREPANCIES FOUND</div>',
                            unsafe_allow_html=True,
                        )
                        for d in discrepancies:
                            sev = d.get("severity", "info").upper()
                            st.markdown(
                                f'<div style="border:2px solid #FF3333;padding:0.75rem 1rem;'
                                f'margin-bottom:0.4rem;font-size:0.82rem;">'
                                f'<span style="color:#FF3333;font-weight:700;">{d["field"].upper()}</span>'
                                f' <span style="color:#666;font-size:0.7rem;">[{sev}]</span>'
                                f'<br><span style="color:#F0F0F0;">{d["detail"]}</span></div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            '<div style="border:2px solid #00FF88;padding:1rem;color:#00FF88;'
                            'font-size:0.82rem;font-weight:700;letter-spacing:0.06em;">'
                            '✓ NO DISCREPANCIES FOUND</div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    err = str(e)
                    if "no extraction" in err.lower() or "409" in err:
                        st.markdown(
                            '<div style="border:2px solid #FF8C00;padding:1rem;color:#FF8C00;font-size:0.82rem;">'
                            'ONE OR MORE SELECTED INVOICES HAVE NO EXTRACTION. '
                            'GO TO EXTRACT PAGE AND RUN EXTRACTION FIRST.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(err)
