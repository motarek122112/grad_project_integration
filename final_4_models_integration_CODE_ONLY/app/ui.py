from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = PROJECT_ROOT / "assets" / "styles.css"


def load_global_css() -> None:
    """Load the website stylesheet."""
    if not STYLE_PATH.exists():
        return

    css_content = STYLE_PATH.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True,
    )


def _profile_image_data_uri(
    image_path: str | None,
) -> str | None:
    """Convert a local profile image into a browser data URI."""
    if not image_path:
        return None

    path = Path(image_path)

    if not path.exists() or not path.is_file():
        return None

    try:
        image_bytes = path.read_bytes()
    except OSError:
        return None

    if not image_bytes:
        return None

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    mime_type = mime_types.get(
        path.suffix.lower(),
        "image/jpeg",
    )

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("ascii")

    return f"data:{mime_type};base64,{encoded_image}"


def render_sidebar_user_card(
    user: dict[str, Any],
) -> None:
    """Render the authenticated user's sidebar card."""
    display_name = (
        str(user.get("full_name") or "").strip()
        or "Ishara User"
    )

    safe_display_name = html.escape(display_name)

    age = user.get("age")
    gender = str(user.get("gender") or "").strip()

    if age is not None and gender:
        profile_line = f"{age} years · {gender}"
    else:
        profile_line = "Profile incomplete"

    safe_profile_line = html.escape(profile_line)

    profile_image_uri = _profile_image_data_uri(
        user.get("profile_image_path")
    )

    if profile_image_uri:
        avatar_html = (
            f'<img src="{profile_image_uri}" '
            f'alt="{safe_display_name} profile picture" '
            'class="profile-avatar-image">'
        )
    else:
        first_letter = html.escape(
            display_name[:1].upper()
        )

        avatar_html = (
            '<div class="avatar-circle">'
            f"{first_letter}"
            "</div>"
        )

    sidebar_html = (
        '<div class="sidebar-user-card">'
        f"{avatar_html}"
        '<div class="sidebar-user-details">'
        f"<strong>{safe_display_name}</strong>"
        f"<span>{safe_profile_line}</span>"
        "</div>"
        "</div>"
    )

    st.sidebar.markdown(
        sidebar_html,
        unsafe_allow_html=True,
    )


def page_header(
    *,
    eyebrow: str,
    title: str,
    description: str,
    icon: str,
) -> None:
    """Render a standard page heading."""
    safe_eyebrow = html.escape(str(eyebrow))
    safe_title = html.escape(str(title))
    safe_description = html.escape(str(description))
    safe_icon = html.escape(str(icon))

    heading_html = (
        '<section class="page-heading">'
        f'<div class="page-heading-icon">{safe_icon}</div>'
        "<div>"
        f'<span class="eyebrow">{safe_eyebrow}</span>'
        f"<h1>{safe_title}</h1>"
        f"<p>{safe_description}</p>"
        "</div>"
        "</section>"
    )

    st.markdown(
        heading_html,
        unsafe_allow_html=True,
    )


def status_badge(
    label: str,
    ready: bool,
) -> None:
    """Render a ready or pending status badge."""
    css_class = (
        "status-ready"
        if ready
        else "status-pending"
    )

    safe_label = html.escape(str(label))

    badge_html = (
        f'<span class="status-badge {css_class}">'
        f"● {safe_label}"
        "</span>"
    )

    st.markdown(
        badge_html,
        unsafe_allow_html=True,
    )


def model_card(
    *,
    icon: str,
    title: str,
    description: str,
    input_label: str,
    output_label: str,
) -> None:
    """Render one model card on the home page."""
    safe_icon = html.escape(str(icon))
    safe_title = html.escape(str(title))
    safe_description = html.escape(str(description))
    safe_input = html.escape(str(input_label))
    safe_output = html.escape(str(output_label))

    card_html = (
        '<article class="model-card">'
        f'<div class="model-card-icon">{safe_icon}</div>'
        f"<h3>{safe_title}</h3>"
        f"<p>{safe_description}</p>"
        '<div class="model-meta">'
        "<span>"
        "<b>Input</b>"
        f"{safe_input}"
        "</span>"
        "<span>"
        "<b>Output</b>"
        f"{safe_output}"
        "</span>"
        "</div>"
        "</article>"
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )


def require_complete_profile() -> bool:
    """Check whether age and gender have been saved."""
    user = st.session_state.get("user", {})

    complete = (
        user.get("age") is not None
        and bool(user.get("gender"))
    )

    if not complete:
        st.warning(
            "Complete your age and gender in "
            "**My Profile** before using the AI models."
        )

    return complete


def render_adapter_help(
    adapter_file: str,
    function_name: str,
) -> None:
    """Display model adapter instructions."""
    with st.expander("Model connection instructions"):
        st.markdown(
            (
                "The website page is ready, but the existing model "
                "inference function must be connected in:\n\n"
                f"`{adapter_file}`\n\n"
                "Replace the placeholder implementation of:\n\n"
                f"`{function_name}`\n\n"
                "Keep the page and authentication code unchanged."
            )
        )