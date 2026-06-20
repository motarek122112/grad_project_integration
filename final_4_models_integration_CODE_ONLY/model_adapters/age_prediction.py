from __future__ import annotations

from pathlib import Path
from typing import Any

from model_adapters.age_gender import predict_age_gender


def predict_age(image_path: Path) -> dict[str, Any]:
    result = predict_age_gender(image_path)

    probabilities = result.get(
        "age_bin_probabilities",
        {},
    )

    confidence = None

    if probabilities:
        confidence = max(
            float(value)
            for value in probabilities.values()
        )

    return {
        "age": int(round(float(result["age"]))),
        "age_group": (
            result.get("age_bin_label")
            or result.get("age_group")
            or "Unknown"
        ),
        "confidence": confidence,
        "face_detected": result.get("face_detected"),
        "bbox": result.get("bbox"),
    }