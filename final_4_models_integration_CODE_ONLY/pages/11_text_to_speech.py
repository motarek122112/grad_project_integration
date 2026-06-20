from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.title("🗣️ Text to Speech")
st.caption("Profile-based voice: gender + age tone, loaded silently in the background.")

user = st.session_state.get("user", {}) or {}
profile_age = user.get("age")
profile_gender = user.get("gender")

st.info(f"Voice profile → Gender: {profile_gender or 'Unknown'} | Age: {profile_age or 'Unknown'}")

text = st.text_area("Enter text", height=150, placeholder="Write text to convert into speech...")

with st.expander("Voice settings", expanded=False):
    use_profile = st.checkbox("Use saved profile age and gender", value=True)
    c1, c2 = st.columns(2)
    with c1:
        manual_gender = st.selectbox("Gender", ["Female", "Male", "Unknown"], index=1 if str(profile_gender).lower() == "male" else 0, disabled=use_profile)
    with c2:
        try:
            default_age = int(float(str(profile_age))) if profile_age is not None else 30
        except Exception:
            default_age = 30
        manual_age = st.number_input("Age", min_value=1, max_value=100, value=default_age, disabled=use_profile)
    voice_override = st.text_input("Optional voice override", placeholder="af_heart, am_michael, ...")
    use_speed_override = st.checkbox("Manual speed override", value=False)
    speed_override = st.slider("Speed", 0.70, 1.30, 1.00, 0.01, disabled=not use_speed_override)


def _run_tts(text: str, gender: Optional[str], age: Optional[Any], voice: Optional[str], speed: Optional[float]) -> Dict[str, Any]:
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tts_{abs(hash((text, gender, age, voice, speed)))}.wav"
    try:
        try:
            from model_adapters.text_to_speech import synthesize_speech
            return synthesize_speech(text=text, gender=gender, age=age, voice=voice, speed=speed)
        except Exception:
            from tts_inference import text_to_speech, get_age_group, get_speed_for_age, get_voice_for_gender_and_age
            selected_voice = voice or get_voice_for_gender_and_age(gender, age)
            selected_speed = float(speed) if speed is not None else get_speed_for_age(age)
            text_to_speech(text, output_path=str(out_path), gender=gender, age=age, voice=selected_voice, speed=selected_speed)
            return {"ok": True, "audio_path": str(out_path), "voice": selected_voice, "speed": selected_speed, "age_group": get_age_group(age), "gender": gender, "age": age}
    except Exception as exc:
        import traceback
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)}

if st.button("Generate Speech", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please write text first.")
        st.stop()
    gender = profile_gender if use_profile else manual_gender
    age = profile_age if use_profile else manual_age
    voice = voice_override.strip() or None
    speed = float(speed_override) if use_speed_override else None
    with st.spinner("Generating speech..."):
        result = _run_tts(text.strip(), gender, age, voice, speed)
    if not result.get("ok"):
        st.error(result.get("error", "TTS failed."))
        with st.expander("Traceback", expanded=False):
            st.code(result.get("traceback", ""))
        st.stop()
    st.success("Speech generated.")
    audio_path = result.get("audio_path") or result.get("output_path")
    if audio_path:
        st.audio(audio_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Voice", result.get("voice", "---"))
    c2.metric("Speed", f"{float(result.get('speed') or 0):.2f}x")
    c3.metric("Age group", result.get("age_group", "---"))
    if audio_path:
        with open(audio_path, "rb") as f:
            st.download_button("Download WAV", f, file_name="ishara_tts.wav", mime="audio/wav", use_container_width=True)
