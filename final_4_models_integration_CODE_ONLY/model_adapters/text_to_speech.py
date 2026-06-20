from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_age(age: Optional[Union[int, float, str]]) -> Optional[Union[int, float, str]]:
    if age is None:
        return None

    value = str(age).strip()

    if not value:
        return None

    return value


def _safe_gender(gender: Optional[str]) -> Optional[str]:
    if gender is None:
        return None

    value = str(gender).strip()

    if not value:
        return None

    return value


@st.cache_resource(show_spinner="Loading Kokoro TTS...")
def load_tts_backend():
    """
    Import the updated Kokoro-based TTS backend once.
    The real pipeline is cached inside tts_inference.py as well.
    """

    from tts_inference import (
        text_to_speech,
        get_age_group,
        get_speed_for_age,
        get_voice_for_gender_and_age,
    )

    return {
        "text_to_speech": text_to_speech,
        "get_age_group": get_age_group,
        "get_speed_for_age": get_speed_for_age,
        "get_voice_for_gender_and_age": get_voice_for_gender_and_age,
    }


def synthesize_speech(
    text: str,
    gender: Optional[str] = None,
    age: Optional[Union[int, float, str]] = None,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Streamlit adapter for Text → Speech.

    Uses profile gender + age to select Kokoro voice and speed.
    """

    try:
        clean_text = " ".join(str(text or "").split())

        if not clean_text:
            return {
                "ok": False,
                "error": "Input text is empty.",
            }

        gender = _safe_gender(gender)
        age = _safe_age(age)

        backend = load_tts_backend()

        age_group = backend["get_age_group"](age)
        selected_voice = voice or backend["get_voice_for_gender_and_age"](gender, age)
        selected_speed = float(speed) if speed is not None else backend["get_speed_for_age"](age)

        output_path = OUTPUTS_DIR / f"tts_streamlit_{st.session_state.get('user', {}).get('id', 'user')}_{abs(hash(clean_text))}.wav"

        backend["text_to_speech"](
            text=clean_text,
            output_path=str(output_path),
            gender=gender,
            age=age,
            voice=selected_voice,
            speed=selected_speed,
        )

        return {
            "ok": True,
            "text": clean_text,
            "audio_path": str(output_path),
            "voice": selected_voice,
            "speed": selected_speed,
            "gender": gender,
            "age": age,
            "age_group": age_group,
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


# Compatibility aliases
text_to_speech = synthesize_speech
generate_speech = synthesize_speech
run_text_to_speech = synthesize_speech
synthesize_text_to_speech = synthesize_speech