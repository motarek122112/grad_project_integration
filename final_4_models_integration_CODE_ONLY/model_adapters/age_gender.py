from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from interfaces.age_gender_interface import AgeGenderInterface


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def find_model(*names: str) -> Path:
    for name in names:
        model_path = MODELS_DIR / name

        if model_path.exists():
            return model_path

    raise FileNotFoundError(
        "Could not find any model file from: "
        + ", ".join(names)
    )


@st.cache_resource(
    show_spinner="Loading age and gender models..."
)
def load_age_gender_model() -> AgeGenderInterface:
    age_model_path = find_model(
        "best_age_efficientnet_b4_finetuned.pth",
        "best_age_efficientnet_b4_finetuned.pth.zip",
    )

    gender_model_path = find_model(
        "best_gender_utkface.pth",
        "best_gender_utkface.pth.zip",
    )

    face_detector_path = find_model(
        "yolov8n-face-lindevs.pt",
        "yolov8n-face-lindevs.pt.zip",
    )

    return AgeGenderInterface(
        age_checkpoint_path=str(age_model_path),
        gender_checkpoint_path=str(gender_model_path),
        face_detector_path=str(face_detector_path),
        use_face_detection=True,
    )


def predict_age_gender(
    image_path: Path,
) -> dict[str, Any]:
    model = load_age_gender_model()

    # One call runs face detection, age prediction,
    # and gender prediction using the same image.
    return model.predict(str(image_path))