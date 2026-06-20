import os
import sys
import shutil
import traceback
import cv2
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


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

    # Old TTS paths are kept for compatibility only.
    # If you replaced TTS with Kokoro inside interfaces/tts_interface.py,
    # these files are not required anymore.
    "tts_acoustic": Path(os.getenv("TTS_ACOUSTIC_CHECKPOINT_PATH", str(MODELS_DIR / "best_tts_acoustic.zip"))),
    "tts_vocoder": Path(os.getenv("VOCODER_CHECKPOINT_PATH", str(MODELS_DIR / "vocoder_best.pt"))),

    "speech": Path(os.getenv("SPEECH_CHECKPOINT_PATH", str(MODELS_DIR / "best_speech_model.pth.zip"))),
    "face_detector": Path(os.getenv("FACE_DETECTOR_PATH", str(MODELS_DIR / "yolov8n-face-lindevs.pt.zip"))),
    "age": Path(os.getenv("AGE_CHECKPOINT_PATH", str(MODELS_DIR / "best_age_efficientnet_b4_finetuned.pth.zip"))),
    "gender": Path(os.getenv("GENDER_CHECKPOINT_PATH", str(MODELS_DIR / "best_gender_utkface.pth.zip"))),
}


# ===============================
# App
# ===============================

app = FastAPI(
    title="Graduation Project - 4 Models Integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

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

    This function rewrites the video as MP4 with stable 30 FPS.
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
            "traceback": traceback.format_exc(limit=5),
        },
    )


def _require_file(path: Path, label: str, optional: bool = False) -> Optional[Path]:
    if path and path.exists():
        return path

    if optional:
        return None

    raise FileNotFoundError(f"{label} not found at: {path}")


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
            result["full_sentence"] = full_sentence

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
    """
    Pick the best segmentation result.

    Prefer:
    1. more detected words
    2. more detected segments
    3. more accepted segments
    4. higher average confidence
    """

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


# ===============================
# Model Loaders
# ===============================

def get_sign_model():
    if "sign" not in _cache:
        from interfaces.sign_language_interface import load_sign_language_interface

        _cache["sign"] = load_sign_language_interface(
            checkpoint_path=str(_require_file(MODEL_PATHS["sign"], "Sign checkpoint")),
            device=DEVICE,
        )

    return _cache["sign"]


def get_language_decoder():
    if "language_decoder" not in _cache:
        from backend.language_decoder import load_language_decoder

        _cache["language_decoder"] = load_language_decoder(
            corpus_path=str(CORPUS_PATH)
        )

    return _cache["language_decoder"]


def get_tts_model():
    if "tts" not in _cache:
        from interfaces.tts_interface import load_tts_interface

        # Compatible with Kokoro wrapper.
        # The Kokoro tts_interface ignores old acoustic/vocoder paths.
        _cache["tts"] = load_tts_interface(
            device=DEVICE,
        )

    return _cache["tts"]


def get_speech_model():
    if "speech" not in _cache:
        from interfaces.speech_interface import load_speech_interface

        _cache["speech"] = load_speech_interface(
            checkpoint_path=str(_require_file(MODEL_PATHS["speech"], "Speech checkpoint")),
            device=DEVICE,
        )

    return _cache["speech"]


def get_age_gender_model():
    if "age_gender" not in _cache:
        from interfaces.age_gender_interface import load_age_gender_interface

        face_detector = _require_file(
            MODEL_PATHS["face_detector"],
            "YOLO face detector",
            optional=True
        )

        _cache["age_gender"] = load_age_gender_interface(
            age_checkpoint_path=str(_require_file(MODEL_PATHS["age"], "Age checkpoint")),
            gender_checkpoint_path=str(_require_file(MODEL_PATHS["gender"], "Gender checkpoint")),
            face_detector_path=str(face_detector) if face_detector else None,
            device=DEVICE,
        )

    return _cache["age_gender"]


# ===============================
# Routes
# ===============================

@app.get("/")
def home():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/status")
def status():
    import importlib.util

    packages = [
        "torch",
        "cv2",
        "mediapipe",
        "ultralytics",
        "timm",
        "librosa",
        "soundfile",
        "kokoro",
    ]

    return {
        "ok": True,
        "device": DEVICE,
        "models": {
            k: {
                "path": str(v),
                "exists": v.exists(),
                "size_mb": round(v.stat().st_size / 1024 / 1024, 2) if v.exists() else None,
            }
            for k, v in MODEL_PATHS.items()
        },
        "resources": {
            "sentence_corpus": {
                "path": str(CORPUS_PATH),
                "exists": CORPUS_PATH.exists(),
            }
        },
        "loaded_interfaces": list(_cache.keys()),
        "packages": {
            p: importlib.util.find_spec(p) is not None
            for p in packages
        },
    }


# ===============================
# 1) Single Sign Video → Text
# ===============================

@app.post("/api/sign/single")
def sign_single(
    video: UploadFile = File(...),
    top_k: int = Form(5),
):
    try:
        raw_video_path = _save_upload(video, "sign_single")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        result = get_sign_model().predict(
            str(video_path),
            top_k=int(top_k)
        )

        result["raw_uploaded_video_path"] = str(raw_video_path)
        result["normalized_video_path"] = str(video_path)

        return {
            "ok": True,
            "result": _json_safe(result)
        }

    except Exception as e:
        return _error_response(e)


# ===============================
# 2) Multi-Sign Video → Sentence
# ===============================

@app.post("/api/sign/sentence")
def sign_sentence(
    video: UploadFile = File(...),
    top_k: int = Form(5),

    # Better defaults for webcam multi-sign videos
    threshold: float = Form(0.08),
    min_pause_sec: float = Form(0.35),
    min_segment_sec: float = Form(0.30),
    confidence_threshold: float = Form(0.08),

    use_language_decoder: bool = Form(True),
):
    try:
        raw_video_path = _save_upload(video, "sign_sentence")

        # Important:
        # Convert webm / webcam videos to stable 30fps mp4.
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        # Try multiple segmentation settings automatically.
        presets = [
            {
                "threshold": float(threshold),
                "min_pause_sec": float(min_pause_sec),
                "min_segment_sec": float(min_segment_sec),
                "confidence_threshold": float(confidence_threshold),
            },
            {
                "threshold": 0.05,
                "min_pause_sec": 0.25,
                "min_segment_sec": 0.20,
                "confidence_threshold": 0.05,
            },
            {
                "threshold": 0.06,
                "min_pause_sec": 0.30,
                "min_segment_sec": 0.25,
                "confidence_threshold": 0.06,
            },
            {
                "threshold": 0.08,
                "min_pause_sec": 0.35,
                "min_segment_sec": 0.30,
                "confidence_threshold": 0.08,
            },
            {
                "threshold": 0.10,
                "min_pause_sec": 0.45,
                "min_segment_sec": 0.35,
                "confidence_threshold": 0.08,
            },
            {
                "threshold": 0.12,
                "min_pause_sec": 0.50,
                "min_segment_sec": 0.35,
                "confidence_threshold": 0.08,
            },
        ]

        best_result = None
        best_score = -1.0

        for preset in presets:
            result = get_sign_model().predict_sentence(
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


# ===============================
# 3) Text → Speech
# ===============================

@app.post("/api/tts")
def text_to_speech(
    text: str = Form(...),
):
    try:
        out_path = OUTPUTS_DIR / f"tts_{_timestamp()}.wav"

        result = get_tts_model().synthesize(
            text=text,
            output_path=str(out_path)
        )

        result["audio_url"] = _url_for_output(result.get("output_path"))
        result["metadata_url"] = _url_for_output(result.get("metadata_path"))

        return {
            "ok": True,
            "result": _json_safe(result)
        }

    except Exception as e:
        return _error_response(e)


# ===============================
# 4) Sign Video → Text → Speech
# Supports both:
# /api/sign/to-speech
# /api/sign/tts
# ===============================

@app.post("/api/sign/to-speech")
@app.post("/api/sign/tts")
def sign_to_speech(
    video: UploadFile = File(...),

    # False = single sign mode
    # True = multi-sign sentence mode
    sentence_mode: bool = Form(False),

    use_language_decoder: bool = Form(True),
):
    try:
        raw_video_path = _save_upload(video, "sign_to_speech")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        sign_model = get_sign_model()

        if sentence_mode:
            sign_result = sign_model.predict_sentence(
                str(video_path),
                top_k=5,
                threshold=0.08,
                min_pause_sec=0.35,
                min_segment_sec=0.30,
                confidence_threshold=0.08,
            )

            sign_result = _apply_sentence_fallback(sign_result)

            if use_language_decoder:
                try:
                    sign_result = get_language_decoder().decode(sign_result)
                except Exception as decoder_error:
                    sign_result["lm_decoder_used"] = False
                    sign_result["lm_error"] = str(decoder_error)

        else:
            sign_result = sign_model.predict(
                str(video_path),
                top_k=5
            )

        text = (
            sign_result.get("lm_sentence")
            or sign_result.get("full_sentence")
            or sign_result.get("text")
            or sign_result.get("clean_text")
            or sign_result.get("word")
            or sign_result.get("gloss")
            or ""
        )

        if not text:
            raise ValueError("No text detected from sign video, so TTS cannot run.")

        out_path = OUTPUTS_DIR / f"sign_tts_{_timestamp()}.wav"

        tts_result = get_tts_model().synthesize(
            text=text,
            output_path=str(out_path)
        )

        tts_result["audio_url"] = _url_for_output(tts_result.get("output_path"))
        tts_result["metadata_url"] = _url_for_output(tts_result.get("metadata_path"))

        merged_result = {
            "pipeline": "sign_language_to_text_to_speech",
            "raw_uploaded_video_path": str(raw_video_path),
            "normalized_video_path": str(video_path),
            "text": text,
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


# ===============================
# 5) Speech → Text
# ===============================

@app.post("/api/speech")
def speech_to_text(
    audio: UploadFile = File(...),
    decoder: str = Form("beam"),
):
    try:
        audio_path = _save_upload(audio, "speech")

        result = get_speech_model().predict(
            str(audio_path),
            use_beam=(decoder != "greedy")
        )

        return {
            "ok": True,
            "result": _json_safe(result)
        }

    except Exception as e:
        return _error_response(e)


# ===============================
# 6) Age & Gender
# ===============================

@app.post("/api/age-gender")
def age_gender(
    image: UploadFile = File(...),
):
    try:
        image_path = _save_upload(image, "age_gender")

        annotated = OUTPUTS_DIR / f"age_gender_{_timestamp()}.jpg"
        cropped = OUTPUTS_DIR / f"face_crop_{_timestamp()}.jpg"

        result = get_age_gender_model().predict_and_save(
            image_path=str(image_path),
            annotated_output_path=str(annotated),
            cropped_face_output_path=str(cropped),
        )

        result["annotated_image_url"] = _url_for_output(str(annotated))
        result["cropped_face_url"] = _url_for_output(str(cropped))

        return {
            "ok": True,
            "result": _json_safe(result)
        }

    except Exception as e:
        return _error_response(e)


# ===============================
# Run
# ===============================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )