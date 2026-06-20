from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.feedback_store import init_feedback_db, save_feedback, list_feedback

init_feedback_db()

st.title("💬 Feedback")
st.caption("Share feedback about Ishara. This helps the developer improve the demo and user experience.")

user = st.session_state.get("user", {}) or {}
user_id = str(user.get("id") or user.get("email") or "demo_user")

with st.form("feedback_form", clear_on_submit=True):
    page = st.selectbox(
        "Which part are you reviewing?",
        ["Overall App", "Live Sign Translation", "Emergency Mode", "Speech to Text", "Text to Speech", "Age Prediction", "Gender Detection"],
    )
    rating = st.slider("Rating", 1, 5, 5)
    comment = st.text_area("Your feedback", placeholder="Write what worked well or what should be improved...")
    submitted = st.form_submit_button("Submit Feedback", type="primary", use_container_width=True)

if submitted:
    save_feedback(
        user_id=user_id,
        page=page,
        rating=int(rating),
        comment=comment.strip(),
        metadata={"profile_age": user.get("age"), "profile_gender": user.get("gender")},
    )
    st.success("Feedback saved. Thank you.")

with st.expander("My recent feedback", expanded=False):
    rows = list_feedback(user_id=user_id, limit=20)
    if not rows:
        st.info("No feedback yet.")
    else:
        for row in rows:
            st.markdown(f"**{row.get('page')}** — {row.get('rating')}/5")
            st.caption(row.get("created_at"))
            st.write(row.get("comment") or "—")
            st.divider()
