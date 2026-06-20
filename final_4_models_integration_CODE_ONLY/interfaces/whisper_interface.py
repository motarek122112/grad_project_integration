from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


class WhisperSpeechToTextInterface:
    """Inference interface for the merged fine-tuned Whisper checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Whisper checkpoint not found: {self.checkpoint_path}"
            )

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "The Whisper checkpoint must contain a dictionary."
            )

        self.base_model = checkpoint.get(
            "base_model",
            "openai/whisper-small",
        )

        self.sample_rate = int(
            checkpoint.get("sample_rate", 16000)
        )

        model_config = checkpoint.get("model_config")

        if model_config is None:
            raise KeyError(
                "The checkpoint does not contain 'model_config'."
            )

        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        if not isinstance(model_config, dict):
            raise TypeError(
                "'model_config' must be a dictionary."
            )

        config = WhisperConfig.from_dict(model_config)

        self.model = WhisperForConditionalGeneration(config)

        state_dict = checkpoint.get("model_state_dict")

        if state_dict is None:
            raise KeyError(
                "The checkpoint does not contain "
                "'model_state_dict'."
            )

        try:
            self.model.load_state_dict(
                state_dict,
                strict=True,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "The saved Whisper weights do not match "
                "the checkpoint model configuration."
            ) from error

        generation_config = checkpoint.get(
            "generation_config",
            {},
        )

        if hasattr(generation_config, "to_dict"):
            generation_config = (
                generation_config.to_dict()
            )

        self.generation_settings: dict[str, Any] = {}

        if isinstance(generation_config, dict):
            self.generation_settings = {
                key: value
                for key, value in generation_config.items()
                if not key.startswith("_")
            }

            try:
                self.model.generation_config.update(
                    **self.generation_settings
                )
            except Exception:
                # The model can still generate using its
                # default generation configuration.
                pass

        # The checkpoint contains the model weights and configuration,
        # but the processor vocabulary and feature extractor are loaded
        # from the recorded base model name.
        self.processor = WhisperProcessor.from_pretrained(
            self.base_model
        )

        self.model.to(self.device)
        self.model.eval()

    def transcribe(
        self,
        audio_path: str | Path,
        max_duration_seconds: float = 30.0,
    ) -> dict[str, Any]:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        audio, sampling_rate = librosa.load(
            str(audio_path),
            sr=self.sample_rate,
            mono=True,
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            raise ValueError(
                "The audio recording is empty."
            )

        if not np.isfinite(audio).all():
            raise ValueError(
                "The audio contains invalid values."
            )

        original_duration = (
            float(len(audio)) / float(sampling_rate)
        )

        maximum_samples = int(
            max_duration_seconds * sampling_rate
        )

        was_trimmed = len(audio) > maximum_samples

        if was_trimmed:
            audio = audio[:maximum_samples]

        processed_duration = (
            float(len(audio)) / float(sampling_rate)
        )

        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(
            self.device,
            dtype=self.model.dtype,
        )

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features=input_features,
                max_new_tokens=225,
            )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()

        configured_language = (
            self.generation_settings.get("language")
        )

        configured_task = (
            self.generation_settings.get(
                "task",
                "transcribe",
            )
        )

        return {
            "text": text,
            "language": configured_language,
            "task": configured_task,
            "duration": original_duration,
            "processed_duration": processed_duration,
            "trimmed": was_trimmed,
            "sample_rate": sampling_rate,
            "device": str(self.device),
            "base_model": self.base_model,
        }

    def get_model_info(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "base_model": self.base_model,
            "sample_rate": self.sample_rate,
            "device": str(self.device),
            "model_class": type(self.model).__name__,
        }