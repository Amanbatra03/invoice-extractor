import os
import streamlit as st
from supabase import create_client

_COOKIE_TOKEN = "inv_token"
_COOKIE_EMAIL = "inv_email"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]
    return create_client(url, key)


_LOGIN_CSS = """
<style>
.brut-login-title {
    font-family: 'Space Grotesk', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #F5F500;
    text-align: center;
    margin-bottom: 0.2rem;
}
.brut-login-sub {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
</style>
"""


def login_page(controller=None) -> bool:
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            '<div class="brut-login-title">INVOICE ANALYST</div>'
            '<div class="brut-login-sub">AI-POWERED INVOICE EXTRACTION &amp; ANALYSIS</div>',
            unsafe_allow_html=True,
        )

        sign_in_tab, sign_up_tab = st.tabs(["SIGN IN", "SIGN UP"])

        with sign_in_tab:
            email = st.text_input("EMAIL", key="signin_email", placeholder="you@company.com")
            password = st.text_input("PASSWORD", type="password", key="signin_pwd")
            if st.button("SIGN IN", type="primary", use_container_width=True, key="signin_btn"):
                if not email.strip() or not password.strip():
                    st.error("ENTER YOUR EMAIL AND PASSWORD.")
                else:
                    try:
                        sb = get_supabase_client()
                        result = sb.auth.sign_in_with_password({"email": email, "password": password})
                        token = result.session.access_token
                        user_email = result.user.email
                        st.session_state["access_token"] = token
                        st.session_state["user_email"] = user_email
                        if controller:
                            controller.set(_COOKIE_TOKEN, token, max_age=_COOKIE_MAX_AGE)
                            controller.set(_COOKIE_EMAIL, user_email, max_age=_COOKIE_MAX_AGE)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"LOGIN FAILED: {exc}")

        with sign_up_tab:
            new_email = st.text_input("EMAIL", key="signup_email", placeholder="you@company.com")
            new_pwd = st.text_input("PASSWORD", type="password", key="signup_pwd",
                                    help="At least 6 characters")
            new_pwd2 = st.text_input("CONFIRM PASSWORD", type="password", key="signup_pwd2")
            if st.button("CREATE ACCOUNT", type="primary", use_container_width=True, key="signup_btn"):
                if not new_email.strip() or not new_pwd.strip():
                    st.error("FILL IN ALL FIELDS.")
                elif new_pwd != new_pwd2:
                    st.error("PASSWORDS DON'T MATCH.")
                elif len(new_pwd) < 6:
                    st.error("PASSWORD MUST BE AT LEAST 6 CHARACTERS.")
                else:
                    try:
                        sb = get_supabase_client()
                        sb.auth.sign_up({"email": new_email, "password": new_pwd})
                        st.success("ACCOUNT CREATED — CHECK YOUR EMAIL TO CONFIRM, THEN SIGN IN.")
                    except Exception as exc:
                        st.error(f"SIGN-UP FAILED: {exc}")

    return False


def is_authenticated() -> bool:
    return bool(st.session_state.get("access_token"))


def get_token() -> str:
    return st.session_state.get("access_token", "")


def logout(controller=None):
    if controller:
        controller.remove(_COOKIE_TOKEN)
        controller.remove(_COOKIE_EMAIL)
    for key in ["access_token", "user_email"]:
        st.session_state.pop(key, None)
    st.rerun()
