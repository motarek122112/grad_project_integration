from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf


def clean_text(text: str) -> str:
    text = str(text or "").strip()
    return " ".join(text.split())


class TTSInterface:
    """Kokoro TTS wrapper with gender-based voice selection."""

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.engine_name = "kokoro"

    def synthesize(
        self,
        text: str,
        output_path: str,
        gender: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        text = clean_text(text)

        if not text:
            raise ValueError("Input text is empty.")

        output_path = str(output_path)

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        from tts_inference import (
            get_voice_for_gender,
            normalize_gender,
            text_to_speech,
        )

        selected_voice = (
            voice
            if voice
            else get_voice_for_gender(gender)
        )

        normalized_gender = normalize_gender(
            gender
        )

        generated_path = text_to_speech(
            text=text,
            output_path=output_path,
            gender=normalized_gender,
            voice=selected_voice,
            speed=speed,
        )

        duration_seconds = None
        sample_rate = 24000

        try:
            audio_info = sf.info(generated_path)

            sample_rate = int(
                audio_info.samplerate
            )

            duration_seconds = round(
                float(audio_info.frames)
                / float(audio_info.samplerate),
                3,
            )
        except Exception:
            pass

        metadata = {
            "input_text": text,
            "clean_text": text,
            "output_path": str(generated_path),
            "sample_rate": sample_rate,
            "duration_sec": duration_seconds,
            "tts_engine": self.engine_name,
            "vocoder": "kokoro",
            "profile_gender": normalized_gender,
            "selected_voice": selected_voice,
            "speed": float(speed),
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        metadata_path = str(
            Path(generated_path).with_suffix(
                ".json"
            )
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                metadata,
                metadata_file,
                ensure_ascii=False,
                indent=2,
            )

        metadata["metadata_path"] = metadata_path

        return metadata


def load_tts_interface(
    acoustic_checkpoint_path: Optional[str] = None,
    vocoder_checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
) -> TTSInterface:
    return TTSInterface(device=device)