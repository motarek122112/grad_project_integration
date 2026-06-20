from __future__ import annotations

import os
import sys
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict

import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UPLOADS_DIR = ROOT / "uploads" / "live_sign"
OUTPUTS_DIR = ROOT / "outputs"
SAMPLES_DIR = ROOT / "samples"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Ishara Sign Live API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

SIGN_API_WARMED_UP = False
SIGN_API_WARMUP_ERROR: Optional[str] = None
TTS_API_WARMED_UP = False
TTS_API_WARMUP_ERROR: Optional[str] = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


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
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if torch is not None and hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    return obj


def _error_response(exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        },
    )


def _save_upload(file: UploadFile, subdir: str = "chunks") -> Path:
    ext = Path(file.filename or "input.webm").suffix or ".webm"
    out_dir = UPLOADS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_timestamp()}{ext}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


def _normalize_video_for_model(input_path: Path, target_fps: int = 30, max_width: int = 640) -> Path:
    """
    Browser webm chunks may report bad FPS to OpenCV.
    Re-write to stable mp4 before the sign model sees it.
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
        if frame is None:
            continue

        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (max_width, int(h * scale)))
        frames.append(frame)

    cap.release()

    if not frames:
        return input_path

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(target_fps), (w, h))

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    return output_path


def _output_url(path: str | Path | None) -> Optional[str]:
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


def _quick_hand_visible_check(video_path: Path, max_checked_frames: int = 45, sample_every: int = 2) -> Dict[str, Any]:
    """
    Optional release-to-confirm helper.
    Returns hand_visible=False only when no left/right hand appears in sampled frames.
    """
    try:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
    except Exception as exc:
        raise ImportError("MediaPipe legacy solutions are required for hand visibility check.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"hand_visible": False, "checked_frames": 0, "hand_frames": 0}

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
        min_tracking_confidence=0.25,
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
                return {"hand_visible": True, "checked_frames": checked_frames, "hand_frames": hand_frames}

    cap.release()
    return {"hand_visible": False, "checked_frames": checked_frames, "hand_frames": hand_frames}


def warmup_sign_model_once() -> None:
    """Preload the updated Sign model and optionally warm MediaPipe with samples/warmup_sign.mp4."""
    global SIGN_API_WARMED_UP, SIGN_API_WARMUP_ERROR

    if SIGN_API_WARMED_UP:
        return

    try:
        from model_adapters.sign_language import load_updated_sign_model, predict_single_sign

        load_updated_sign_model()

        warmup_video = SAMPLES_DIR / "warmup_sign.mp4"
        if warmup_video.exists():
            predict_single_sign(video_path=warmup_video, top_k=1)
            print("[SIGN LIVE API] Full warm-up done using samples/warmup_sign.mp4")
        else:
            print("[SIGN LIVE API] Model loaded. No warmup video found at samples/warmup_sign.mp4")

        SIGN_API_WARMED_UP = True
        SIGN_API_WARMUP_ERROR = None

    except Exception as exc:
        SIGN_API_WARMUP_ERROR = str(exc)
        print("[SIGN LIVE API] Warm-up failed:")
        print(traceback.format_exc())


def warmup_tts_once() -> None:
    """Preload Kokoro TTS and generate one tiny audio file before the first Speak Sentence click."""
    global TTS_API_WARMED_UP, TTS_API_WARMUP_ERROR

    if TTS_API_WARMED_UP:
        return

    try:
        from tts_inference import text_to_speech as kokoro_text_to_speech
        out_path = OUTPUTS_DIR / "_warmup_live_tts.wav"

        try:
            from tts_inference import get_pipeline
            get_pipeline("a")
        except Exception:
            pass

        try:
            kokoro_text_to_speech("Hello", output_path=str(out_path), gender="Male", age=34)
        except TypeError:
            try:
                kokoro_text_to_speech("Hello", str(out_path), gender="Male")
            except TypeError:
                kokoro_text_to_speech("Hello", str(out_path))

        TTS_API_WARMED_UP = True
        TTS_API_WARMUP_ERROR = None
        print("[SIGN LIVE API] TTS warm-up done.")

    except Exception as exc:
        TTS_API_WARMUP_ERROR = str(exc)
        print("[SIGN LIVE API] TTS warm-up failed:")
        print(traceback.format_exc())


@app.on_event("startup")
def startup_warmup():
    warmup_sign_model_once()
    warmup_tts_once()


@app.get("/api/sign-v2/warmup-status")
def warmup_status():
    return {
        "ok": SIGN_API_WARMED_UP and TTS_API_WARMED_UP,
        "sign_warmed_up": SIGN_API_WARMED_UP,
        "sign_error": SIGN_API_WARMUP_ERROR,
        "tts_warmed_up": TTS_API_WARMED_UP,
        "tts_error": TTS_API_WARMUP_ERROR,
    }


@app.post("/api/sign-v2/single")
def sign_single(
    video: UploadFile = File(...),
    top_k: int = Form(5),
    check_hand: bool = Form(False),
):
    try:
        raw_video_path = _save_upload(video, "single")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        if check_hand:
            hand_check = _quick_hand_visible_check(video_path)
            if not hand_check.get("hand_visible"):
                return {
                    "ok": True,
                    "result": {
                        "no_hand": True,
                        "hand_visible": False,
                        "message": "No hand detected in chunk.",
                        "hand_check": hand_check,
                        "top_k": [],
                    },
                }

        from model_adapters.sign_language import predict_single_sign
        result = predict_single_sign(video_path=video_path, top_k=int(top_k))
        result["raw_uploaded_video_path"] = str(raw_video_path)
        result["normalized_video_path"] = str(video_path)
        return {"ok": bool(result.get("ok", True)), "result": _json_safe(result)}

    except Exception as exc:
        return _error_response(exc)


@app.post("/api/sign-v2/sentence")
def sign_sentence(
    video: UploadFile = File(...),
    top_k: int = Form(5),
    threshold: float = Form(0.08),
    min_pause_sec: float = Form(0.35),
    min_segment_sec: float = Form(0.30),
    confidence_threshold: float = Form(0.08),
    use_language_decoder: bool = Form(True),
):
    try:
        raw_video_path = _save_upload(video, "sentence")
        video_path = _normalize_video_for_model(raw_video_path, target_fps=30)

        from model_adapters.sign_language import predict_sign_sentence
        result = predict_sign_sentence(
            video_path=video_path,
            top_k=int(top_k),
            threshold=float(threshold),
            min_pause_sec=float(min_pause_sec),
            min_segment_sec=float(min_segment_sec),
            confidence_threshold=float(confidence_threshold),
            use_language_decoder=bool(use_language_decoder),
        )
        result["raw_uploaded_video_path"] = str(raw_video_path)
        result["normalized_video_path"] = str(video_path)
        return {"ok": bool(result.get("ok", True)), "result": _json_safe(result)}

    except Exception as exc:
        return _error_response(exc)


@app.post("/api/tts")
def text_to_speech(
    text: str = Form(...),
    gender: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    speed: Optional[str] = Form(None),
):
    try:
        from tts_inference import (
            text_to_speech as kokoro_text_to_speech,
            get_age_group,
            get_speed_for_age,
            get_voice_for_gender_and_age,
        )

        clean_text = " ".join(str(text or "").split())
        if not clean_text:
            raise ValueError("Input text is empty.")

        selected_speed = float(speed) if speed not in (None, "") else get_speed_for_age(age)
        selected_voice = voice if voice else get_voice_for_gender_and_age(gender, age)
        age_group = get_age_group(age)

        out_path = OUTPUTS_DIR / f"live_tts_{_timestamp()}.wav"
        kokoro_text_to_speech(
            text=clean_text,
            output_path=str(out_path),
            gender=gender,
            age=age,
            voice=selected_voice,
            speed=selected_speed,
        )

        return {
            "ok": True,
            "result": {
                "text": clean_text,
                "output_path": str(out_path),
                "audio_url": _output_url(out_path),
                "gender": gender,
                "age": age,
                "age_group": age_group,
                "voice": selected_voice,
                "speed": selected_speed,
            },
        }

    except Exception as exc:
        return _error_response(exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sign_live_api:app", host="127.0.0.1", port=8000, reload=False)
