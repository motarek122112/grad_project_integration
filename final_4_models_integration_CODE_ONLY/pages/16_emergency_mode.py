from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.title("🚨 Emergency Mode")
st.caption("Fast phrases for urgent communication. Select a phrase and Ishara speaks it aloud.")

user = st.session_state.get("user", {}) or {}
profile_age = user.get("age")
profile_gender = user.get("gender")

st.info(f"Voice profile → Gender: {profile_gender or 'Unknown'} | Age: {profile_age or 'Unknown'}")

phrases: List[Dict[str, str]] = [
    {"en": "I need help", "ar": "أحتاج مساعدة"},
    {"en": "Call a doctor", "ar": "اتصل بالطبيب"},
    {"en": "I am in pain", "ar": "أنا أتألم"},
    {"en": "Where is the hospital?", "ar": "أين المستشفى؟"},
    {"en": "Call my family", "ar": "اتصل بعائلتي"},
]

language = st.radio("Phrase display", ["English", "Arabic"], horizontal=True)


def _speak_phrase(text: str):
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"emergency_{abs(hash((text, profile_gender, profile_age)))}.wav"
    try:
        from tts_inference import text_to_speech, get_voice_for_gender_and_age, get_speed_for_age, get_age_group
        voice = get_voice_for_gender_and_age(profile_gender, profile_age)
        speed = get_speed_for_age(profile_age)
        age_group = get_age_group(profile_age)
        text_to_speech(text, output_path=str(out_path), gender=profile_gender, age=profile_age, voice=voice, speed=speed)
        return {"ok": True, "audio_path": str(out_path), "voice": voice, "speed": speed, "age_group": age_group}
    except Exception as exc:
        import traceback
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)}

cols = st.columns(2)
for i, phrase in enumerate(phrases):
    display = phrase["ar"] if language == "Arabic" else phrase["en"]
    speak_text = phrase["en"]
    with cols[i % 2]:
        if st.button(display, key=f"emergency_{i}", type="primary", use_container_width=True):
            with st.spinner("Speaking..."):
                result = _speak_phrase(speak_text)
            if not result.get("ok"):
                st.error(result.get("error", "TTS failed."))
                with st.expander("Traceback", expanded=False):
                    st.code(result.get("traceback", ""))
            else:
                st.success(speak_text)
                st.audio(result["audio_path"])

st.divider()
custom = st.text_input("Custom phrase", placeholder="Type a sentence to speak quickly...")
if st.button("Speak custom phrase", use_container_width=True):
    if not custom.strip():
        st.warning("Write a phrase first.")
    else:
        with st.spinner("Speaking..."):
            result = _speak_phrase(custom.strip())
        if not result.get("ok"):
            st.error(result.get("error", "TTS failed."))
        else:
            st.audio(result["audio_path"])
