from __future__ import annotations

from pathlib import Path
from typing import Any

from model_adapters.age_gender import predict_age_gender


def predict_gender(image_path: Path) -> dict[str, Any]:
    result = predict_age_gender(image_path)

    predicted_label = str(result.get("gender", "Unknown")).capitalize()
    positive_label = str(
        result.get("gender_positive_label", "female")
    ).capitalize()
    positive_probability = float(
        result.get("gender_probability_positive", 0.5)
    )

    if positive_label.lower() == "female":
        probabilities = {
            "Female": positive_probability,
            "Male": 1.0 - positive_probability,
        }
    else:
        probabilities = {
            "Male": positive_probability,
            "Female": 1.0 - positive_probability,
        }

    return {
        "gender": predicted_label,
        "confidence": result.get("gender_confidence"),
        "probabilities": probabilities,
        "face_detected": result.get("face_detected"),
        "bbox": result.get("bbox"),
    }
