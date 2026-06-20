from __future__ import annotations

import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PAGES_DIR = PROJECT_ROOT / "pages"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.auth import (
    create_account,
    login_user,
    logout_user,
    restore_authenticated_user,
)
from app.database import init_database
from app.ui import load_global_css, render_sidebar_user_card
from app.fast_preload import start_fast_preload
from app.auth import (
    create_account,
    login_user,
    logout_user,
    restore_authenticated_user,
)
from app.database import init_database
from app.ui import load_global_css, render_sidebar_user_card


st.set_page_config(
    page_title="Ishara AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_database()
load_global_css()
restore_authenticated_user()


def authentication_page() -> None:
    """Render the login and signup experience."""
    left, center, right = st.columns([1, 1.35, 1])

    with center:
        st.markdown(
            """
            <div class="auth-brand">
                <div class="brand-badge">🤟</div>
                <h1>Ishara</h1>
                <p>AI-powered accessible communication</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        login_tab, signup_tab = st.tabs(["Log in", "Create account"])

        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                st.subheader("Welcome back")
                email = st.text_input(
                    "Email address",
                    placeholder="name@example.com",
                    key="login_email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password",
                )
                submitted = st.form_submit_button(
                    "Log in",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                success, message = login_user(email=email, password=password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with signup_tab:
            with st.form("signup_form", clear_on_submit=False):
                st.subheader("Create your Ishara account")
                full_name = st.text_input(
                    "Full name",
                    placeholder="Your full name",
                    key="signup_name",
                )
                email = st.text_input(
                    "Email address",
                    placeholder="name@example.com",
                    key="signup_email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="At least 8 characters",
                    key="signup_password",
                )
                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="Repeat your password",
                    key="signup_confirm_password",
                )
                accepted = st.checkbox(
                    "I agree to use this application responsibly.",
                    key="signup_terms",
                )
                submitted = st.form_submit_button(
                    "Create account",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                success, message = create_account(
                    full_name=full_name,
                    email=email,
                    password=password,
                    confirm_password=confirm_password,
                    accepted_terms=accepted,
                )
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.caption(
            "Passwords are stored as salted PBKDF2 hashes. "
            "The local SQLite database is intended for the graduation-project demo."
        )


def logout_page() -> None:
    logout_user()
    st.rerun()


if not st.session_state.get("authenticated", False):
    page = st.navigation(
        [
            st.Page(
                authentication_page,
                title="Log in",
                icon=":material/login:",
                default=True,
            )
        ],
        position="hidden",
    )
    page.run()
else:
    user = st.session_state["user"]

    profile_complete = (
        user.get("age") is not None
        and bool(user.get("gender"))
    )

    render_sidebar_user_card(user)

    # Mandatory onboarding immediately after login.
    if not profile_complete:
        st.sidebar.info(
            "Take one face photo to complete your age and gender profile."
        )

        pages = {
            "Profile Setup": [
                st.Page(
                    PAGES_DIR / "01_profile.py",
                    title="Complete AI Profile",
                    icon=":material/account_circle:",
                    default=True,
                ),st.Page(
                    logout_page,
                    title="Log out",
                    icon=":material/logout:",
                ),
            ],
        }

    else:
        pages = {
            "Main": [
                st.Page(
                    PAGES_DIR / "00_home.py",
                    title="Home",
                    icon=":material/home:",
                    default=True,
                ),
                st.Page(
                    PAGES_DIR / "01_profile.py",
                    title="My Profile",
                    icon=":material/account_circle:",
                ),
            ],
            "AI Models": [
                st.Page(
                    PAGES_DIR / "10_sign_language.py",
                    title="Sign Language to Text",
                    icon=":material/sign_language:",
                ),
                st.Page(
                    PAGES_DIR / "15_live_sign_translation.py",
                    title="Live Sign Translation",
                    icon=":material/videocam:",
                ),st.Page(
                    PAGES_DIR / "11_text_to_speech.py",
                    title="Text to Speech",
                    icon=":material/record_voice_over:",
                ),
                st.Page(
                    PAGES_DIR / "12_speech_to_text.py",
                    title="Speech to Text",
                    icon=":material/hearing:",
                ),
                st.Page(
                    PAGES_DIR / "13_age_prediction.py",
                    title="Age Prediction",
                    icon=":material/cake:",
                ),
                st.Page(
                    PAGES_DIR / "14_gender_detection.py",
                    title="Gender Detection",
                    icon=":material/person_search:",
                ),
                st.Page(
                    PAGES_DIR / "16_emergency_mode.py",
                    title="Emergency Mode",
                    icon=":material/emergency:",
                ),
            ],
            "Project": [
                st.Page(
                    PAGES_DIR / "90_about.py",
                    title="About Ishara",
                    icon=":material/info:",
                ),st.Page(
                    logout_page,
                    title="Log out",
                    icon=":material/logout:",
                ),
            ],
        }

    selected_page = st.navigation(
        pages,
        position="sidebar",
        expanded=True,
    )

    selected_page.run()