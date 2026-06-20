from __future__ import annotations

import sys
import time
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_adapters.sign_language import predict_single_sign
from model_adapters.text_to_speech import synthesize_speech
from model_adapters.speech_to_text import transcribe_audio

try:
    from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
except Exception:  # pragma: no cover
    WebRtcMode = None
    VideoProcessorBase = object
    webrtc_streamer = None


# ================================
# Page state
# ================================

def _init_state() -> None:
    defaults = {
        "live_words": [],
        "live_glosses": [],
        "live_last_text": "---",
        "live_last_gloss": "---",
        "live_last_confidence": 0.0,
        "live_last_prediction_at": 0.0,
        "live_last_audio": None,
        "live_stt_text": "",
        "live_status": "Open the camera, then sign clearly in front of it.",
        "live_busy": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ================================
# Helpers
# ================================

def _user_profile() -> Dict[str, Any]:
    return st.session_state.get("user", {}) or {}


def _sentence() -> str:
    words = [str(w).strip() for w in st.session_state.live_words if str(w).strip()]
    return " ".join(words).strip()


def _append_prediction(text: str, gloss: str, confidence: float, repeat_cooldown_sec: float) -> bool:
    text = str(text or "").strip()
    gloss = str(gloss or "").strip()
    if not text and gloss:
        text = gloss
    if not text:
        return False

    now = time.time()
    last_word = st.session_state.live_words[-1] if st.session_state.live_words else None
    last_at = float(st.session_state.get("live_last_prediction_at", 0.0) or 0.0)

    # Avoid repeated same sign when the hand stays in front of the camera.
    if last_word and last_word.lower() == text.lower() and (now - last_at) < repeat_cooldown_sec:
        st.session_state.live_status = "Same sign ignored to avoid repetition."
        return False

    st.session_state.live_words.append(text)
    st.session_state.live_glosses.append(gloss or text)
    st.session_state.live_last_prediction_at = now
    st.session_state.live_status = "Sign accepted."
    return True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _frames_to_video(frames: List[np.ndarray], fps: int = 12) -> Optional[Path]:
    if not frames:
        return None

    clean_frames: List[np.ndarray] = []
    for frame in frames:
        if frame is None:
            continue
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            continue
        clean_frames.append(arr)

    if not clean_frames:
        return None

    height, width = clean_frames[0].shape[:2]
    temp_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name)
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        float(fps),
        (int(width), int(height)),
    )

    for frame in clean_frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    return temp_path


def _pick_prediction_text(result: Dict[str, Any]) -> Tuple[str, str, float]:
    text = str(result.get("text") or result.get("sentence") or result.get("word") or "").strip()
    gloss = str(result.get("gloss") or result.get("label") or "").strip()
    confidence = _safe_float(result.get("confidence"), 0.0)

    top_k = result.get("top_k") or []
    if isinstance(top_k, list) and top_k:
        top1 = top_k[0] if isinstance(top_k[0], dict) else {}
        text = text or str(top1.get("text") or top1.get("word") or top1.get("gloss") or top1.get("label") or "").strip()
        gloss = gloss or str(top1.get("gloss") or top1.get("label") or text or "").strip()
        confidence = confidence or _safe_float(top1.get("confidence") or top1.get("score") or top1.get("probability"), 0.0)

    return text, gloss, confidence


# ================================
# WebRTC processor
# ================================

class LiveSignVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: deque[Tuple[float, np.ndarray]] = deque(maxlen=180)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        with self.lock:
            self.frames.append((now, img.copy()))
        return frame

    def get_recent_frames(self, seconds: float = 1.25, max_frames: int = 36) -> List[np.ndarray]:
        now = time.time()
        with self.lock:
            selected = [img for ts, img in list(self.frames) if now - ts <= seconds]

        if not selected:
            return []

        # Downsample to keep inference fast.
        if len(selected) > max_frames:
            idx = np.linspace(0, len(selected) - 1, max_frames).astype(int)
            selected = [selected[i] for i in idx]

        return selected


# ================================
# UI
# ================================

st.title("🎥 Live Sign Translation")
st.caption("Online Streamlit live mode — no separate FastAPI server required.")

st.info("This page is deployment-friendly: camera capture, sign prediction, speech output, and STT run inside Streamlit.")

with st.expander("Live settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        clip_seconds = st.slider("Clip seconds", 0.60, 2.50, 1.20, 0.10)
        top_k = st.slider("Top-K", 1, 10, 5)
    with c2:
        min_confidence = st.slider("Minimum confidence", 0.00, 0.95, 0.30, 0.05)
        repeat_cooldown_sec = st.slider("Repeat cooldown", 0.50, 5.00, 2.50, 0.25)
    with c3:
        max_frames = st.slider("Max frames per prediction", 12, 64, 36, 4)
        auto_refresh_hint = st.checkbox("Show simple interface", value=True)

if webrtc_streamer is None:
    st.error("streamlit-webrtc is not installed. Add streamlit-webrtc and av to requirements.txt, then redeploy.")
    st.code("streamlit-webrtc>=0.72,<1\nav>=12", language="text")
    st.stop()

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Camera")
    webrtc_ctx = webrtc_streamer(
        key="ishara-online-live-sign",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveSignVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    b1, b2, b3, b4 = st.columns(4)

    predict_now = b1.button("Predict Sign", type="primary", use_container_width=True)
    undo = b2.button("Undo", use_container_width=True)
    clear = b3.button("Clear", use_container_width=True)
    speak = b4.button("Speak", use_container_width=True)

    if undo and st.session_state.live_words:
        st.session_state.live_words.pop()
        if st.session_state.live_glosses:
            st.session_state.live_glosses.pop()
        st.session_state.live_status = "Last sign removed."

    if clear:
        st.session_state.live_words = []
        st.session_state.live_glosses = []
        st.session_state.live_last_text = "---"
        st.session_state.live_last_gloss = "---"
        st.session_state.live_last_confidence = 0.0
        st.session_state.live_last_audio = None
        st.session_state.live_status = "Sentence cleared."

    if predict_now:
        if not webrtc_ctx.video_processor:
            st.warning("Open the camera first.")
        else:
            frames = webrtc_ctx.video_processor.get_recent_frames(seconds=float(clip_seconds), max_frames=int(max_frames))
            if not frames:
                st.warning("No camera frames yet. Wait one second and try again.")
            else:
                video_path = _frames_to_video(frames, fps=12)
                if video_path is None:
                    st.error("Could not create a temporary video clip.")
                else:
                    with st.spinner("Predicting current sign..."):
                        result = predict_single_sign(video_path=video_path, top_k=int(top_k))
                    try:
                        video_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                    if not result.get("ok"):
                        st.error(result.get("error", "Prediction failed."))
                        with st.expander("Debug", expanded=False):
                            st.json(result)
                    else:
                        text, gloss, conf = _pick_prediction_text(result)
                        st.session_state.live_last_text = text or "---"
                        st.session_state.live_last_gloss = gloss or text or "---"
                        st.session_state.live_last_confidence = conf

                        if conf >= float(min_confidence):
                            _append_prediction(text, gloss, conf, repeat_cooldown_sec=float(repeat_cooldown_sec))
                        else:
                            st.session_state.live_status = "Prediction ignored because confidence is below threshold."

    if speak:
        sentence = _sentence()
        if not sentence:
            st.warning("No sentence to speak yet.")
        else:
            profile = _user_profile()
            gender = profile.get("gender")
            age = profile.get("age")
            with st.spinner("Generating speech..."):
                tts_result = synthesize_speech(sentence, gender=gender, age=age)
            if not tts_result.get("ok"):
                st.error(tts_result.get("error", "TTS failed."))
            else:
                st.session_state.live_last_audio = tts_result.get("audio_path")
                st.success("Speech generated.")

with right:
    st.subheader("Current Output")
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:16px;background:#f8fafc;border:1px solid #e5e7eb;margin-bottom:12px;">
            <div style="font-size:13px;color:#64748b;margin-bottom:4px;">Current prediction</div>
            <div style="font-size:30px;font-weight:800;color:#0f172a;">{st.session_state.live_last_text}</div>
            <div style="font-size:13px;color:#64748b;margin-top:8px;">Confidence: {float(st.session_state.live_last_confidence or 0):.3f}</div>
        </div>
        <div style="padding:18px;border-radius:16px;background:#ffffff;border:1px solid #e5e7eb;margin-bottom:12px;">
            <div style="font-size:13px;color:#64748b;margin-bottom:4px;">Gloss sequence</div>
            <div style="font-size:18px;font-weight:700;color:#0f172a;">{' + '.join(st.session_state.live_glosses) if st.session_state.live_glosses else '---'}</div>
        </div>
        <div style="padding:18px;border-radius:16px;background:#ffffff;border:1px solid #e5e7eb;">
            <div style="font-size:13px;color:#64748b;margin-bottom:4px;">English sentence</div>
            <div style="font-size:24px;font-weight:800;color:#0f172a;">{_sentence() or '---'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(st.session_state.live_status)

    if st.session_state.live_last_audio:
        st.audio(st.session_state.live_last_audio)


st.divider()
st.subheader("🗣️ Speech to Text for hearing speaker")
st.caption("The hearing person can record or upload audio, and the deaf user reads the transcription.")

recorded_audio = None
if hasattr(st, "audio_input"):
    recorded_audio = st.audio_input("Record speech")

uploaded_audio = st.file_uploader(
    "Or upload an audio file",
    type=["wav", "mp3", "m4a", "webm", "ogg", "flac"],
    key="live_stt_upload_online",
)

audio_source = recorded_audio or uploaded_audio

if audio_source is not None:
    st.audio(audio_source)

    if st.button("Convert Speech to Text", type="primary", use_container_width=True):
        suffix = Path(getattr(audio_source, "name", "speech.webm")).suffix or ".webm"
        temp_audio = Path(tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name)
        temp_audio.write_bytes(audio_source.getvalue())

        with st.spinner("Transcribing speech..."):
            stt_result = transcribe_audio(audio_path=temp_audio)

        try:
            temp_audio.unlink(missing_ok=True)
        except Exception:
            pass

        if not stt_result.get("ok"):
            st.error(stt_result.get("error", "Speech to Text failed."))
            with st.expander("Debug", expanded=False):
                st.json(stt_result)
        else:
            text = stt_result.get("text") or stt_result.get("transcription") or stt_result.get("sentence") or ""
            st.session_state.live_stt_text = text
            st.success("Transcription ready.")

if st.session_state.live_stt_text:
    st.markdown(
        f"""
        <div style="padding:20px;border-radius:16px;background:#f8fafc;border:1px solid #e5e7eb;">
            <div style="font-size:13px;color:#64748b;margin-bottom:6px;">Speech transcription</div>
            <div style="font-size:30px;font-weight:800;color:#0f172a;">{st.session_state.live_stt_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
