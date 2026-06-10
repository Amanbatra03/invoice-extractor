import hmac
import os

import streamlit as st


def check_password() -> bool:
    """Gate the app behind APP_PASSWORD when set; open when unset (dev mode)."""
    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        return True
    if st.session_state.get("auth_ok"):
        return True

    def _verify():
        if hmac.compare_digest(st.session_state.get("password", ""), expected):
            st.session_state["auth_ok"] = True
            del st.session_state["password"]   # don't keep the secret around
        else:
            st.session_state["auth_ok"] = False

    st.text_input("Password", type="password", key="password", on_change=_verify)
    if st.session_state.get("auth_ok") is False:
        st.error("Incorrect password.")
    return False
