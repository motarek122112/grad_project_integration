from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import soundfile as sf


_PIPELINES: Dict[str, object] = {}

# ── Voice options per gender ───────────────────────────────────────────────────
# Kokoro built-in voices — these are all available out of the box, no extra model needed
VOICES = {
    "Female": {
        "young":  "af_heart",      # bright, energetic (0-25)
        "adult":  "af_bella",      # warm, clear (26-50)
        "senior": "af_sarah",      # calm, measured (51+)
    },
    "Male": {
        "young":  "am_michael",    # bright, energetic (0-25)
        "adult":  "am_adam",       # deep, clear (26-50)
        "senior": "am_echo",       # slow, calm (51+)
    },
}

# Fallback if a specific voice is unavailable
FEMALE_VOICE = "af_heart"
MALE_VOICE   = "am_michael"


# ── Age group helpers ──────────────────────────────────────────────────────────
def get_age_group(age: Optional[Union[int, float, str]]) -> str:
    """
    Map a numeric age (from the age prediction model) to young / adult / senior.
    Returns 'adult' when age is None or unparseable.
    """
    try:
        age_int = int(float(str(age)))
    except (TypeError, ValueError):
        return "adult"

    if age_int <= 25:
        return "young"
    if age_int <= 50:
        return "adult"
    return "senior"


def get_speed_for_age(age: Optional[Union[int, float, str]]) -> float:
    """
    Slightly adjust speaking rate based on age.
    Young  → 1.10 (a little faster / more energetic)
    Adult  → 1.00 (neutral)
    Senior → 0.88 (a little slower / more deliberate)
    """
    group = get_age_group(age)
    return {"young": 1.10, "adult": 1.00, "senior": 0.88}[group]


# ── Gender helpers ─────────────────────────────────────────────────────────────
def normalize_gender(gender: Optional[str]) -> str:
    value = str(gender or "").strip().lower()
    if value == "male":
        return "Male"
    if value == "female":
        return "Female"
    return "Unknown"


def get_voice_for_gender_and_age(
    gender: Optional[str],
    age: Optional[Union[int, float, str]] = None,
) -> str:
    """
    Pick the best Kokoro voice for the combination of gender + age group.
    Falls back to gender-only selection when age is missing.
    """
    normalized_gender = normalize_gender(gender)
    age_group         = get_age_group(age)

    gender_key = normalized_gender if normalized_gender in VOICES else "Female"
    voice      = VOICES[gender_key].get(age_group, VOICES[gender_key]["adult"])
    return voice


# ── Main function ──────────────────────────────────────────────────────────────
def get_pipeline(lang_code: str = "a"):
    """Load and cache one Kokoro pipeline per language code."""
    if lang_code not in _PIPELINES:
        from kokoro import KPipeline
        _PIPELINES[lang_code] = KPipeline(lang_code=lang_code)
    return _PIPELINES[lang_code]


def text_to_speech(
    text: str,
    output_path: str = "output.wav",
    gender: Optional[str] = None,
    age: Optional[Union[int, float, str]] = None,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
) -> str:
    """
    Convert text to speech using Kokoro, adapting voice and speed
    to the user's gender and age from their profile.

    Parameters
    ----------
    text        : text to synthesize
    output_path : where to save the .wav file
    gender      : "Male" / "Female" / None  — from the age-gender model
    age         : numeric age or None        — from the age prediction model
    voice       : override voice name (optional, skips auto-selection)
    speed       : override speed (optional, skips auto-selection)

    Returns
    -------
    output_path (str)
    """
    clean_text = " ".join(str(text or "").split())
    if not clean_text:
        raise ValueError("Input text is empty.")

    # Voice selection — explicit override wins, otherwise use gender + age
    selected_voice = voice if voice else get_voice_for_gender_and_age(gender, age)

    # Speed selection — explicit override wins, otherwise derive from age
    selected_speed = float(speed) if speed is not None else get_speed_for_age(age)

    pipeline     = get_pipeline(lang_code="a")
    audio_chunks = []

    for _, _, audio in pipeline(
        clean_text,
        voice=selected_voice,
        speed=selected_speed,
    ):
        audio_chunks.append(np.asarray(audio, dtype=np.float32))

    if not audio_chunks:
        raise RuntimeError("Kokoro did not generate any audio.")

    sf.write(output_path, np.concatenate(audio_chunks), 24000)
    return output_path
