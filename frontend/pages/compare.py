import asyncio
import pandas as pd
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.subheader("Compare")
    try:
        invoices_data = asyncio.run(client.list_invoices())
        invoices = [i for i in invoices_data.get("items", []) if i["file_type"] == "pdf"]
    except Exception as e:
        st.error(str(e))
        return
    if len(invoices) < 2:
        st.info("Load at least 2 PDF invoices to compare.")
        return
    selected = [inv["id"] for inv in invoices if st.checkbox(inv["file_name"], key=f"cmp_{inv['id']}")]
    if len(selected) >= 2 and st.button("Compare Selected", type="primary"):
        with st.spinner("Comparing..."):
            try:
                result = asyncio.run(client.compare_invoices(selected))
                table = result.get("table", {})
                discrepancies = result.get("discrepancies", [])
                if table:
                    disc_fields = {d["field"] for d in discrepancies}
                    rows = [{"Field": f, **v} for f, v in table.items()]
                    df = pd.DataFrame(rows).set_index("Field")
                    def highlight(df):
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for f in disc_fields:
                            if f in styles.index:
                                styles.loc[f] = "background-color: #5C3A2B"
                        return styles
                    st.dataframe(df.style.apply(highlight, axis=None), use_container_width=True)
                if discrepancies:
                    st.subheader("Discrepancies")
                    for d in discrepancies:
                        st.warning(f"**{d['field']}** ({d.get('severity','info')}): {d['detail']}")
                else:
                    st.success("No discrepancies found.")
            except Exception as e:
                st.error(str(e))
