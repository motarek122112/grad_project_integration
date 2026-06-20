from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESOURCES_DIR = PROJECT_ROOT / "resources"

SIGN_CHECKPOINT_PATH = Path(
    os.getenv(
        "SIGN_CHECKPOINT_PATH",
        str(MODELS_DIR / "asl_landmark_best_v3.pth"),
    )
)

SENTENCE_CORPUS_PATH = Path(
    os.getenv(
        "SENTENCE_CORPUS_PATH",
        str(RESOURCES_DIR / "sentence_corpus.txt"),
    )
)

DEVICE = os.getenv("DEVICE", "cpu")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner="Loading updated Sign Language model...")
def load_updated_sign_model(checkpoint_path: str = str(SIGN_CHECKPOINT_PATH), device: str = DEVICE):
    """Load Mohamed's updated Sign Language model once for Streamlit."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Updated sign checkpoint not found at: {checkpoint}\n"
            "Put asl_landmark_best_v3.pth inside models/ or set SIGN_CHECKPOINT_PATH."
        )

    from interfaces.sign_language_interface import load_sign_language_interface

    try:
        return load_sign_language_interface(
            checkpoint_path=str(checkpoint),
            device=device,
        )
    except TypeError:
        # Compatibility with older interfaces that accepted checkpoint as first positional argument.
        return load_sign_language_interface(str(checkpoint))


@st.cache_resource(show_spinner=False)
def load_sentence_decoder(corpus_path: str = str(SENTENCE_CORPUS_PATH)):
    """Load optional N-gram language decoder for multi-sign sentence output."""

    corpus = Path(corpus_path)
    if not corpus.exists():
        return None

    try:
        from backend.language_decoder import load_language_decoder
        return load_language_decoder(str(corpus))
    except Exception:
        return None


def json_safe(obj: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if torch is not None and hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    return obj


def clean_repeated_words(text: str) -> str:
    """Remove immediate duplicates like: I I want breakfast -> I want breakfast."""

    words = str(text or "").strip().split()
    cleaned = []
    for word in words:
        if not cleaned or cleaned[-1].lower() != word.lower():
            cleaned.append(word)
    return " ".join(cleaned)


def pick_text(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""

    text = (
        result.get("lm_sentence")
        or result.get("full_sentence")
        or result.get("text")
        or result.get("clean_text")
        or result.get("predicted_text")
        or result.get("word")
        or result.get("gloss")
        or result.get("label")
        or ""
    )

    if not text and result.get("word_sequence"):
        text = " ".join(result.get("word_sequence") or [])

    return clean_repeated_words(str(text).strip())


def standardize_single_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw or {}

    top_k = raw.get("top_k") or raw.get("top_predictions") or []
    top1 = top_k[0] if isinstance(top_k, list) and top_k else {}

    gloss = (
        raw.get("gloss")
        or raw.get("label")
        or raw.get("predicted_label")
        or top1.get("gloss")
        or top1.get("label")
        or ""
    )

    text = pick_text(raw) or top1.get("text") or top1.get("word") or gloss

    confidence = (
        raw.get("confidence")
        or raw.get("probability")
        or raw.get("score")
        or top1.get("confidence")
        or top1.get("probability")
        or top1.get("score")
        or 0.0
    )

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    return {
        "ok": True,
        "mode": "single",
        "text": str(text or "").strip(),
        "gloss": str(gloss or "").strip(),
        "confidence": confidence,
        "top_k": top_k,
        "raw_result": json_safe(raw),
    }


def standardize_sentence_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw or {}
    text = pick_text(raw)

    return {
        "ok": True,
        "mode": "sentence",
        "text": text,
        "sentence": text,
        "word_sequence": raw.get("word_sequence") or [],
        "gloss_sequence": raw.get("gloss_sequence") or [],
        "segments": raw.get("segments") or [],
        "lm_decoder_used": raw.get("lm_decoder_used", False),
        "lm_scores": raw.get("lm_scores"),
        "raw_result": json_safe(raw),
    }


def predict_single_sign(
    video_path: str | Path,
    top_k: int = 5,
    checkpoint_path: str = str(SIGN_CHECKPOINT_PATH),
    device: str = DEVICE,
) -> Dict[str, Any]:
    """Predict one isolated sign video using the updated sign model."""

    try:
        model = load_updated_sign_model(str(checkpoint_path), device)
        raw = model.predict(str(video_path), top_k=int(top_k))
        return standardize_single_result(raw)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def predict_sign_sentence(
    video_path: str | Path,
    top_k: int = 5,
    threshold: float = 0.08,
    min_pause_sec: float = 0.35,
    min_segment_sec: float = 0.30,
    confidence_threshold: float = 0.08,
    use_language_decoder: bool = True,
    checkpoint_path: str = str(SIGN_CHECKPOINT_PATH),
    device: str = DEVICE,
) -> Dict[str, Any]:
    """Predict a controlled multi-sign sentence using the updated sign model."""

    try:
        model = load_updated_sign_model(str(checkpoint_path), device)

        # Some interfaces support predict_sentence; some older ones don't.
        if not hasattr(model, "predict_sentence"):
            single = predict_single_sign(
                video_path=video_path,
                top_k=top_k,
                checkpoint_path=checkpoint_path,
                device=device,
            )
            single["mode"] = "sentence_fallback_single"
            return single

        presets = [
            {
                "threshold": float(threshold),
                "min_pause_sec": float(min_pause_sec),
                "min_segment_sec": float(min_segment_sec),
                "confidence_threshold": float(confidence_threshold),
            },
            {"threshold": 0.05, "min_pause_sec": 0.25, "min_segment_sec": 0.20, "confidence_threshold": 0.05},
            {"threshold": 0.06, "min_pause_sec": 0.30, "min_segment_sec": 0.25, "confidence_threshold": 0.06},
            {"threshold": 0.08, "min_pause_sec": 0.35, "min_segment_sec": 0.30, "confidence_threshold": 0.08},
            {"threshold": 0.10, "min_pause_sec": 0.45, "min_segment_sec": 0.35, "confidence_threshold": 0.08},
        ]

        best_raw = None
        best_score = -1.0

        for preset in presets:
            raw = model.predict_sentence(
                str(video_path),
                top_k=int(top_k),
                threshold=preset["threshold"],
                min_pause_sec=preset["min_pause_sec"],
                min_segment_sec=preset["min_segment_sec"],
                confidence_threshold=preset["confidence_threshold"],
            )

            raw = raw or {}
            raw["selected_segmentation_preset"] = preset

            if use_language_decoder:
                decoder = load_sentence_decoder()
                if decoder is not None:
                    try:
                        raw = decoder.decode(raw)
                    except Exception as decoder_error:
                        raw["lm_decoder_used"] = False
                        raw["lm_error"] = str(decoder_error)

            segments = raw.get("segments") or []
            words = raw.get("word_sequence") or []
            accepted = sum(1 for seg in segments if seg.get("accepted"))
            avg_conf = 0.0
            confs = []
            for seg in segments:
                try:
                    confs.append(float(seg.get("confidence", 0.0)))
                except Exception:
                    pass
            if confs:
                avg_conf = sum(confs) / len(confs)

            score = len(words) * 100.0 + len(segments) * 30.0 + accepted * 15.0 + avg_conf
            raw["selection_score"] = round(float(score), 4)

            if score > best_score:
                best_score = score
                best_raw = raw

        return standardize_sentence_result(best_raw or {})

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def predict_sign_language(
    video_path: str | Path,
    mode: str = "single",
    sentence: Optional[bool] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Universal adapter entry point for the Streamlit Sign Language page."""

    use_sentence = sentence if sentence is not None else str(mode).lower() in {"sentence", "multi", "multi-sign", "multisign"}

    if use_sentence:
        return predict_sign_sentence(video_path, **kwargs)

    return predict_single_sign(video_path, **kwargs)


# Compatibility aliases for different page versions.
predict_sign = predict_sign_language
run_sign_language = predict_sign_language
transcribe_sign = predict_sign_language
analyze_sign_video = predict_sign_language
predict_sentence = predict_sign_sentence
