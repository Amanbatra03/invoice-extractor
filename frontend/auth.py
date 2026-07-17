import os
import streamlit as st
from supabase import create_client


def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]
    return create_client(url, key)


_LOGIN_CSS = """
<style>
.auth-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 70vh;
    padding: 1rem;
}
.auth-card {
    background: rgba(44, 43, 40, 0.60);
    backdrop-filter: blur(28px) saturate(160%);
    -webkit-backdrop-filter: blur(28px) saturate(160%);
    border: 1px solid rgba(255,255,255,0.10);
    border-top-color: rgba(255,255,255,0.22);
    border-radius: 22px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
    padding: 2.6rem 2.8rem 2.2rem;
    width: 100%;
    max-width: 420px;
}
.auth-logo {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.9rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #ECEAE4;
    margin-bottom: 0.2rem;
    text-align: center;
}
.auth-tagline {
    color: #A8A599;
    font-size: 0.88rem;
    text-align: center;
    margin-bottom: 1.8rem;
}
</style>
"""

_LOGIN_OPEN = """
<div class="auth-wrap"><div class="auth-card">
  <div class="auth-logo">Invoice Analyst</div>
  <div class="auth-tagline">AI-powered invoice extraction &amp; analysis</div>
</div></div>
"""


def login_page() -> bool:
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            '<div class="auth-logo" style="text-align:center;font-family:Fraunces,Georgia,serif;'
            'font-size:1.75rem;font-weight:600;letter-spacing:-0.02em;color:#ECEAE4;margin-bottom:0.15rem;">Invoice Analyst</div>'
            '<div style="color:#A8A599;font-size:0.85rem;text-align:center;margin-bottom:1.6rem;">'
            'AI-powered invoice extraction &amp; analysis</div>',
            unsafe_allow_html=True,
        )

        sign_in_tab, sign_up_tab = st.tabs(["Sign In", "Sign Up"])

        with sign_in_tab:
            email = st.text_input("Email", key="signin_email", placeholder="you@company.com")
            password = st.text_input("Password", type="password", key="signin_pwd")
            if st.button("Sign In", type="primary", use_container_width=True, key="signin_btn"):
                if not email.strip() or not password.strip():
                    st.error("Enter your email and password.")
                else:
                    try:
                        sb = get_supabase_client()
                        result = sb.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state["access_token"] = result.session.access_token
                        st.session_state["user_email"] = result.user.email
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Login failed: {exc}")

        with sign_up_tab:
            new_email = st.text_input("Email", key="signup_email", placeholder="you@company.com")
            new_pwd = st.text_input("Password", type="password", key="signup_pwd",
                                    help="At least 6 characters")
            new_pwd2 = st.text_input("Confirm password", type="password", key="signup_pwd2")
            if st.button("Create account", type="primary", use_container_width=True, key="signup_btn"):
                if not new_email.strip() or not new_pwd.strip():
                    st.error("Fill in all fields.")
                elif new_pwd != new_pwd2:
                    st.error("Passwords don't match.")
                elif len(new_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        sb = get_supabase_client()
                        sb.auth.sign_up({"email": new_email, "password": new_pwd})
                        st.success("Account created — check your email to confirm, then sign in.")
                    except Exception as exc:
                        st.error(f"Sign-up failed: {exc}")

    return False


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def get_token() -> str:
    return st.session_state.get("access_token", "")


def logout():
    for key in ["access_token", "user_email"]:
        st.session_state.pop(key, None)
    st.rerun()
