import os
import streamlit as st
from supabase import create_client


def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]
    return create_client(url, key)


def login_page() -> bool:
    st.title("Invoice Analyst")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign In", type="primary"):
        try:
            sb = get_supabase_client()
            result = sb.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state["access_token"] = result.session.access_token
            st.session_state["user_email"] = result.user.email
            st.rerun()
        except Exception as exc:
            st.error(f"Login failed: {exc}")
    return False


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def get_token() -> str:
    return st.session_state.get("access_token", "")


def logout():
    for key in ["access_token", "user_email"]:
        st.session_state.pop(key, None)
    st.rerun()
