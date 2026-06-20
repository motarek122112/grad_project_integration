from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from interfaces.whisper_interface import (
    WhisperSpeechToTextInterface,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_whisper_small_mixed_lora.pth"
)


@st.cache_resource(
    show_spinner="Loading the fine-tuned Whisper model..."
)
def load_whisper_model() -> WhisperSpeechToTextInterface:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "The Whisper checkpoint was not found at:\n"
            f"{CHECKPOINT_PATH}"
        )

    return WhisperSpeechToTextInterface(
        checkpoint_path=CHECKPOINT_PATH,
        device=None,
    )


def transcribe_audio(
    audio_path: Path,
) -> dict[str, Any]:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model = load_whisper_model()

    return model.transcribe(
        audio_path=audio_path,
        max_duration_seconds=30.0,
    )