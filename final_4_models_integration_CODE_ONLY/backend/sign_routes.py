import os
import sys
import shutil
import traceback
import cv2
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse


# ======================================================
# Sign Language Model Routes - Safe Web Merge Version
# ------------------------------------------------------
# Add this file to: backend/sign_routes.py
# Then include it inside your current backend/app.py:
#
#   from backend.sign_routes import router as sign_router
#   app.include_router(sign_router)
#
# This file does NOT replace your existing 4-model backend.
# It only adds sign-language endpoints under /api/sign-v2/*
# and a separate /live page.
# ======================================================

router = APIRouter()


# ===============================
# Paths
# ===============================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"
FRONTEND_DIR = ROOT / "frontend"
MODELS_DIR = ROOT / "models"
RESOURCES_DIR = ROOT / "resources"
CORPUS_PATH = RESOURCES_DIR / "sentence_corpus.txt"

for p in [UPLOADS_DIR, OUTPUTS_DIR, RESOURCES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEVICE = os.getenv("DEVICE", "cpu")

MODEL_PATHS = {
    "sign": Path(os.getenv("SIGN_CHECKPOINT_PATH", str(MODELS_DIR / "asl_landmark_best_v3.pth"))),
}

_cache: Dict[str, Any] = {}


# ===============================
# Helpers
# ===============================

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _save_upload(file: UploadFile, subdir: str) -> Path:
    ext = Path(file.filename or "input.bin").suffix or ".bin"

    out_dir = UPLOADS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{_timestamp()}{ext}"

    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return path


def _normalize_video_for_model(input_path: Path, target_fps: int = 30) -> Path:
    """
    Fix browser-recorded videos, especially .webm files.

    Some webcam recordings are read by OpenCV with wrong FPS,
    for example fps=1000. This breaks pause-based segmentation.
    This function rewrites the video as MP4 with stable FPS.
    """

    input_path = Path(input_path)

    normalized_dir = UPLOADS_DIR / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    output_path = normalized_dir / f"{input_path.stem}_normalized.mp4"

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        return input_path

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is not None:
            frames.append(frame)

    cap.release()

    if not frames:
        return input_path

    h, w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(target_fps),
        (w, h)
    )

    for frame in frames:
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

    writer.release()

    return output_path


def _url_for_output(path: Optional[str]) -> Optional[str]:
    if not path:
        return None

    p = Path(path)

    if not p.is_absolute():
        p = (ROOT / p).resolve()

    try:
        return "/outputs/" + p.relative_to(OUTPUTS_DIR.resolve()).as_posix()

    except Exception:
        if p.exists():
            dst = OUTPUTS_DIR / p.name

            if p.resolve() != dst.resolve():
                shutil.copy2(p, dst)

            return "/outputs/" + dst.name

        return None


def _json_safe(obj: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]

    if np is not None and isinstance(obj, np.generic):
        return obj.item()

    if torch is not None and hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()

    if isinstance(obj, Path):
        return str(obj)

    return obj


def _error_response(exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=6),
        },
    )


def _require_file(path: Path, label: str, optional: bool = False) -> Optional[Path]:
    if path and path.exists():
        return path

    if optional:
        return None

    raise FileNotFoundError(f"{label} not found at: {path}")


def _clean_repeated_words(text: str) -> str:
    """Remove immediate repeated words, e.g. 'I I want breakfast' -> 'I want breakfast'."""

    text = " ".join(str(text or "").strip().split())

    if not text:
        return ""

    words = text.split()
    cleaned = []

    for word in words:
        if not cleaned or cleaned[-1].lower() != word.lower():
            cleaned.append(word)

    return " ".join(cleaned)


def _apply_sentence_fallback(result: dict) -> dict:
    """
    Camera/live videos sometimes produce correct top-1 predictions,
    but the segments are rejected because confidence is low.
    This fallback builds a raw candidate sentence from segment top-1 predictions.
    """

    if not isinstance(result, dict):
        return result

    existing_text = result.get("text", "")

    if existing_text:
        full_sentence = str(existing_text).strip()

        if full_sentence:
            full_sentence = full_sentence[0].upper() + full_sentence[1:]
            result["full_sentence"] = _clean_repeated_words(full_sentence)
            result["text"] = result["full_sentence"]

        return result

    segments = result.get("segments", [])

    if not segments:
        result["full_sentence"] = ""
        return result

    fallback_words = []
    fallback_glosses = []
    fallback_confidences = []

    for seg in segments:
        word = seg.get("text", "")
        gloss = seg.get("gloss", "")
        conf = seg.get("confidence", None)

        if (not word or not gloss) and seg.get("top_k"):
            top1 = seg["top_k"][0]

            if not word:
                word = top1.get("text", "")

            if not gloss:
                gloss = top1.get("gloss", "")

            if conf is None:
                conf = top1.get("confidence", None)

        if word:
            fallback_words.append(word)

        if gloss:
            fallback_glosses.append(gloss)

        if conf is not None:
            try:
                fallback_confidences.append(float(conf))
            except Exception:
                pass

    if fallback_words:
        full_sentence = " ".join(fallback_words).strip()
        full_sentence = full_sentence[0].upper() + full_sentence[1:]
        full_sentence = _clean_repeated_words(full_sentence)

        result["text"] = full_sentence
        result["full_sentence"] = full_sentence
        result["word_sequence"] = fallback_words
        result["gloss_sequence"] = fallback_glosses
        result["used_low_confidence_fallback"] = True

        if fallback_confidences:
            result["average_confidence"] = sum(fallback_confidences) / len(fallback_confidences)

    else:
        result["full_sentence"] = ""

    return result


def _score_sentence_result(result: dict) -> float:
    if not isinstance(result, dict):
        return -1.0

    segments = result.get("segments", []) or []
    words = result.get("word_sequence", []) or []

    segment_count = len(segments)
    word_count = len(words)
    accepted_count = sum(1 for s in segments if s.get("accepted"))

    confs = []

    for s in segments:
        try:
            confs.append(float(s.get("confidence", 0.0)))
        except Exception:
            pass

    avg_conf = sum(confs) / len(confs) if confs else 0.0

    score = (
        word_count * 100.0 +
        segment_count * 30.0 +
        accepted_count * 15.0 +
        avg_conf
    )

    return float(score)


def _get_mediapipe_holistic():
    """Support old MediaPipe solutions import used by this project."""

    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
            return mp.solutions.holistic

    except Exception:
        pass

    try:
        import mediapipe.python.solutions.holistic as mp_holistic
        return mp_holistic

    except Exception as exc:
        raise ImportError(
            "MediaPipe legacy solutions are required. Use Python 3.10 environment with mediapipe==0.10.14."
        ) from exc


def _quick_hand_visible_check(video_path, max_checked_frames=32, sample_every=2):
    """
    Optional live-mode hand visibility check.

    It scans across the chunk, not only the first frames, so it does not
    reject signs that appear after recording starts.
    """

    mp_holistic = _get_mediapipe_holistic()
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "hand_visible": False,
            "checked_frames": 0,
            "hand_frames": 0
        }

    checked_frames = 0
    hand_frames = 0
    frame_index = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25
    ) as holistic:

        while checked_frames < max_checked_frames:
            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            if frame_index % sample_every != 0:
                continue

            checked_frames += 1

            h, w = frame.shape[:2]
            if w > 480:
                scale = 480 / float(w)
                frame = cv2.resize(frame, (480, int(h * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            if result.left_hand_landmarks or result.right_hand_landmarks:
                hand_frames += 1
                cap.release()

                return {
                    "hand_visible": True,
                    "checked_frames": checked_frames,
                    "hand_frames": hand_frames
                }

    cap.release()

    return {
        "hand_visible": False,
        "checked_frames": checked_frames,
        "hand_frames": hand_frames
    }


# ===============================
# Model Loaders
# ===============================

def get_sign_model():
    if "sign_v2" not in _cache:
        from interfaces.sign_language_interface import load_sign_language_interface

        _cache["sign_v2"] = load_sign_language_interface(
            checkpoint_path=str(_require_file(MODEL_PATHS["sign"], "Sign checkpoint")),
            device=DEVICE,
        )

    return _cache["sign_v2"]


def get_language_decoder():
    if "language_decoder_v2" not in _cache:
        from backend.language_decoder import load_language_decoder

        _cache["language_decoder_v2"] = load_language_decoder(
            corpus_path=str(CORPUS_PATH)
        )

    return _cache["language_decoder_v2"]


def get_tts_model():
    """
    Used only for Sign -> Speech pipeline.
    This does not replace your existing TTS endpoint/model.
    """

    if "tts_for_sign_v2" not in _cache:
        from interfaces.tts_interface import load_tts_interface

        _cache["tts_for_sign_v2"] = load_tts_interface(
            device=DEVICE,
        )

    return _cache["tts_for_sign_v2"]


# ===============================
# Page Route
# ===============================

@router.get("/live")
def live_page():
    live_html = FRONTEND_DIR / "live.html"

    if live_html.exists():
        return FileResponse(str(live_html))

    return JSONResponse(
        status_code=404,
        content={
            "ok": False,
            "error": "frontend/live.html not found."
        }
    )


# ===============================
# Sign API v2 Routes
# ===============================

@router.post("/api/sign-v2/single")
def sign_single_v2(
    video: UploadFile = File(...),
    top_k: int = Form(5),
    check_hand: bool = Form(False),
):
    try:
        raw_video_path = _save_upload(video, "sign_v2_single")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        hand_check = None

        if check_hand:
            hand_check = _quick_hand_visible_check(video_path)

            if not hand_check["hand_visible"]:
                return {
                    "ok": True,
                    "result": {
                        "no_hand": True,
                        "hand_visible": False,
                        "message": "No hand detected in frame.",
                        "hand_check": hand_check,
                        "top_k": []
                    }
                }

        result = get_sign_model().predict(
            str(video_path),
            top_k=int(top_k)
        )

        result["raw_uploaded_video_path"] = str(raw_video_path)
        result["normalized_video_path"] = str(video_path)

        if hand_check is not None:
            result["hand_visible"] = True
            result["hand_check"] = hand_check

        return {
            "ok": True,
            "result": _json_safe(result)
        }

    except Exception as e:
        return _error_response(e)


@router.post("/api/sign-v2/sentence")
def sign_sentence_v2(
    video: UploadFile = File(...),
    top_k: int = Form(5),
    threshold: float = Form(0.08),
    min_pause_sec: float = Form(0.35),
    min_segment_sec: float = Form(0.30),
    confidence_threshold: float = Form(0.08),
    use_language_decoder: bool = Form(True),
):
    try:
        raw_video_path = _save_upload(video, "sign_v2_sentence")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

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
            {"threshold": 0.12, "min_pause_sec": 0.50, "min_segment_sec": 0.35, "confidence_threshold": 0.08},
        ]

        best_result = None
        best_score = -1.0

        sign_model = get_sign_model()

        for preset in presets:
            result = sign_model.predict_sentence(
                str(video_path),
                top_k=int(top_k),
                threshold=preset["threshold"],
                min_pause_sec=preset["min_pause_sec"],
                min_segment_sec=preset["min_segment_sec"],
                confidence_threshold=preset["confidence_threshold"],
            )

            result = _apply_sentence_fallback(result)

            if use_language_decoder:
                try:
                    result = get_language_decoder().decode(result)
                except Exception as decoder_error:
                    result["lm_decoder_used"] = False
                    result["lm_error"] = str(decoder_error)

            score = _score_sentence_result(result)

            result["selected_segmentation_preset"] = preset
            result["raw_uploaded_video_path"] = str(raw_video_path)
            result["normalized_video_path"] = str(video_path)
            result["selection_score"] = round(float(score), 4)

            if score > best_score:
                best_score = score
                best_result = result

        return {
            "ok": True,
            "result": _json_safe(best_result)
        }

    except Exception as e:
        return _error_response(e)


@router.post("/api/sign-v2/to-speech")
@router.post("/api/sign-v2/tts")
def sign_to_speech_v2(
    video: UploadFile = File(...),
    sentence_mode: bool = Form(True),
    top_k: int = Form(5),
    threshold: float = Form(0.08),
    min_pause_sec: float = Form(0.35),
    min_segment_sec: float = Form(0.30),
    confidence_threshold: float = Form(0.08),
    use_language_decoder: bool = Form(True),
):
    try:
        raw_video_path = _save_upload(video, "sign_v2_to_speech")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        sign_model = get_sign_model()

        if sentence_mode:
            # Re-use sentence endpoint logic locally to choose the best segmentation.
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

            best_sign_result = None
            best_score = -1.0

            for preset in presets:
                sign_result = sign_model.predict_sentence(
                    str(video_path),
                    top_k=int(top_k),
                    threshold=preset["threshold"],
                    min_pause_sec=preset["min_pause_sec"],
                    min_segment_sec=preset["min_segment_sec"],
                    confidence_threshold=preset["confidence_threshold"],
                )

                sign_result = _apply_sentence_fallback(sign_result)

                if use_language_decoder:
                    try:
                        sign_result = get_language_decoder().decode(sign_result)
                    except Exception as decoder_error:
                        sign_result["lm_decoder_used"] = False
                        sign_result["lm_error"] = str(decoder_error)

                score = _score_sentence_result(sign_result)
                sign_result["selected_segmentation_preset"] = preset
                sign_result["selection_score"] = round(float(score), 4)

                if score > best_score:
                    best_score = score
                    best_sign_result = sign_result

            sign_result = best_sign_result

            text = (
                sign_result.get("lm_sentence")
                or sign_result.get("full_sentence")
                or sign_result.get("text")
                or (
                    " ".join(sign_result.get("word_sequence", []))
                    if sign_result.get("word_sequence")
                    else ""
                )
                or ""
            )

        else:
            sign_result = sign_model.predict(
                str(video_path),
                top_k=int(top_k)
            )

            text = (
                sign_result.get("text")
                or sign_result.get("clean_text")
                or sign_result.get("word")
                or sign_result.get("gloss")
                or ""
            )

        text = _clean_repeated_words(text)

        if not text:
            raise ValueError("No text detected from sign video, so TTS cannot run.")

        out_path = OUTPUTS_DIR / f"sign_sentence_tts_{_timestamp()}.wav"

        tts_result = get_tts_model().synthesize(
            text=text,
            output_path=str(out_path)
        )

        tts_result["audio_url"] = _url_for_output(tts_result.get("output_path"))
        tts_result["metadata_url"] = _url_for_output(tts_result.get("metadata_path"))

        merged_result = {
            "pipeline": "sign_v2_multi_sign_video_to_sentence_to_speech",
            "raw_uploaded_video_path": str(raw_video_path),
            "normalized_video_path": str(video_path),
            "sentence_mode": bool(sentence_mode),
            "text": text,
            "sentence": text,
            "sign_result": _json_safe(sign_result),
            "tts_result": _json_safe(tts_result),
            "audio_url": tts_result.get("audio_url"),
            "metadata_url": tts_result.get("metadata_url"),
        }

        return {
            "ok": True,
            "result": _json_safe(merged_result)
        }

    except Exception as e:
        return _error_response(e)
