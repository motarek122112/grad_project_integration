from __future__ import annotations

import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import av  # noqa: F401
    import cv2
    from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = ""
except Exception as exc:
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = str(exc)

try:
    import mediapipe as mp  # noqa: F401
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_ERROR = ""
except Exception as exc:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_ERROR = str(exc)

from model_adapters.sign_language import predict_single_sign
from model_adapters.text_to_speech import synthesize_speech
from model_adapters.speech_to_text import transcribe_audio

_FRAME_BUFFER: deque = deque(maxlen=180)
_FRAME_LOCK = threading.Lock()


def _init_state() -> None:
    defaults = {
        "auto_live_running": False,
        "sentence_words": [],
        "gloss_sequence": [],
        "last_candidate": "",
        "stable_count": 0,
        "last_accept_time": 0.0,
        "current_prediction": "---",
        "current_confidence": None,
        "last_raw": {},
        "last_error": "",
        "last_audio_path": "",
        "stt_text": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with _FRAME_LOCK:
        _FRAME_BUFFER.append((time.time(), img.copy()))
    return frame


def _get_recent_frames(seconds: float) -> List[Any]:
    now = time.time()
    with _FRAME_LOCK:
        return [img for ts, img in list(_FRAME_BUFFER) if now - ts <= seconds]


def _write_frames_to_video(frames: List[Any], fps: float = 24.0) -> Optional[Path]:
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    out_path = Path(tmp.name)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        return None
    for img in frames:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    return out_path


def _extract_prediction(result: Dict[str, Any]) -> Tuple[str, str, float, Dict[str, Any]]:
    if not isinstance(result, dict):
        return "", "", 0.0, {}
    text = str(result.get("text") or result.get("word") or result.get("sentence") or "").strip()
    gloss = str(result.get("gloss") or result.get("label") or "").strip()
    conf = result.get("confidence", 0.0)
    top_k = result.get("top_k") or result.get("top_predictions") or []
    if isinstance(top_k, list) and top_k:
        top1 = top_k[0] or {}
        text = text or str(top1.get("text") or top1.get("word") or "").strip()
        gloss = gloss or str(top1.get("gloss") or top1.get("label") or "").strip()
        conf = conf or top1.get("confidence") or top1.get("score") or top1.get("probability") or 0.0
    try:
        conf = float(conf or 0.0)
    except Exception:
        conf = 0.0
    display = text or gloss
    return display, gloss or display, conf, result


def _accept_word(word: str, gloss: str, repeat_cooldown_ms: int) -> bool:
    if not word:
        return False
    now = time.time()
    last_time = float(st.session_state.get("last_accept_time") or 0.0)
    if now - last_time < repeat_cooldown_ms / 1000.0:
        return False
    words = st.session_state.get("sentence_words", [])
    if words and str(words[-1]).lower() == str(word).lower():
        return False
    words.append(word)
    st.session_state["sentence_words"] = words
    glosses = st.session_state.get("gloss_sequence", [])
    glosses.append(gloss or word)
    st.session_state["gloss_sequence"] = glosses
    st.session_state["last_accept_time"] = now
    return True


def _process_live_chunk(
    *,
    top_k: int,
    chunk_ms: int,
    live_min_confidence: float,
    release_confirm_confidence: float,
    hold_confirm_confidence: float,
    stable_count_required: int,
    repeat_cooldown_ms: int,
) -> None:
    frames = _get_recent_frames(seconds=max(0.5, chunk_ms / 1000.0))
    if len(frames) < 4:
        st.session_state["last_error"] = "Waiting for enough camera frames..."
        return
    video_path = None
    try:
        fps = min(30, max(12, len(frames) / max(0.5, chunk_ms / 1000.0)))
        video_path = _write_frames_to_video(frames, fps=fps)
        if not video_path:
            st.session_state["last_error"] = "Could not create temporary video chunk."
            return
        result = predict_single_sign(video_path=video_path, top_k=top_k)
        if not result.get("ok", True):
            st.session_state["last_error"] = result.get("error", "Sign prediction failed.")
            st.session_state["last_raw"] = result
            return
        word, gloss, conf, raw = _extract_prediction(result)
        st.session_state["current_prediction"] = word or "---"
        st.session_state["current_confidence"] = conf
        st.session_state["last_raw"] = raw
        st.session_state["last_error"] = ""

        if not word or conf < release_confirm_confidence:
            st.session_state["last_candidate"] = ""
            st.session_state["stable_count"] = 0
            return

        if conf >= hold_confirm_confidence:
            _accept_word(word, gloss, repeat_cooldown_ms)
            st.session_state["last_candidate"] = word
            st.session_state["stable_count"] = stable_count_required
            return

        if conf >= live_min_confidence:
            if st.session_state.get("last_candidate") == word:
                st.session_state["stable_count"] = int(st.session_state.get("stable_count") or 0) + 1
            else:
                st.session_state["last_candidate"] = word
                st.session_state["stable_count"] = 1
            if int(st.session_state["stable_count"]) >= int(stable_count_required):
                _accept_word(word, gloss, repeat_cooldown_ms)
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
    finally:
        try:
            if video_path:
                Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


def _speak_sentence() -> None:
    sentence = " ".join(st.session_state.get("sentence_words", [])).strip()
    if not sentence:
        st.warning("No sentence to speak yet.")
        return
    user = st.session_state.get("user", {}) or {}
    result = synthesize_speech(text=sentence, gender=user.get("gender"), age=user.get("age"))
    if not result.get("ok"):
        st.error(result.get("error", "TTS failed."))
        return
    st.session_state["last_audio_path"] = result.get("audio_path", "")


def _undo_last() -> None:
    words = st.session_state.get("sentence_words", [])
    glosses = st.session_state.get("gloss_sequence", [])
    if words:
        words.pop()
    if glosses:
        glosses.pop()
    st.session_state["sentence_words"] = words
    st.session_state["gloss_sequence"] = glosses


def _clear_sentence() -> None:
    for key, value in {
        "sentence_words": [],
        "gloss_sequence": [],
        "last_candidate": "",
        "stable_count": 0,
        "current_prediction": "---",
        "current_confidence": None,
        "last_raw": {},
        "last_audio_path": "",
    }.items():
        st.session_state[key] = value


def _run_stt(audio_file) -> None:
    if audio_file is None:
        st.warning("Record or upload audio first.")
        return
    suffix = Path(getattr(audio_file, "name", "recording.wav")).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getbuffer())
        audio_path = Path(tmp.name)
    try:
        result = transcribe_audio(audio_path=audio_path)
        if not result.get("ok"):
            st.error(result.get("error", "Speech to text failed."))
            with st.expander("Raw STT result", expanded=False):
                st.json(result)
            return
        st.session_state["stt_text"] = result.get("text") or result.get("transcription") or result.get("sentence") or ""
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass


_init_state()

st.title("🎥 Live Sign Translation")
st.caption("Auto live translation runs inside Streamlit Cloud — no separate FastAPI server required.")

st.markdown(
    """
    <style>
    video {
        width: 100% !important;
        max-height: 620px !important;
        object-fit: cover !important;
        border-radius: 18px !important;
        background: #000 !important;
    }
    .result-card {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(15, 23, 42, 0.20);
        margin-bottom: 14px;
    }
    .big-live-word {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.15;
    }
    .soft-muted {
        opacity: 0.75;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander("Live settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        top_k = st.slider("Top-K", 1, 10, 5)
        chunk_ms = st.slider("Chunk duration ms", 600, 2500, 1000, 100)
    with c2:
        loop_gap_ms = st.slider("Loop gap ms", 100, 1000, 200, 50)
        repeat_cooldown_ms = st.slider("Repeat cooldown ms", 500, 5000, 2500, 100)
    with c3:
        live_min_confidence = st.slider("Min confidence", 0.05, 0.95, 0.30, 0.01)
        hold_confirm_confidence = st.slider("Hold confirm confidence", 0.05, 0.95, 0.62, 0.01)
        release_confirm_confidence = st.slider("Release confidence", 0.01, 0.80, 0.15, 0.01)
    stable_count_required = st.slider("Stable count required", 1, 5, 1, 1)

if not WEBRTC_AVAILABLE:
    st.error("streamlit-webrtc / av is required. Add streamlit-webrtc and av to requirements.txt, then redeploy.")
    st.code("streamlit-webrtc>=0.72,<1\nav>=12", language="text")
    st.caption(WEBRTC_IMPORT_ERROR)
    st.stop()

if not MEDIAPIPE_AVAILABLE:
    st.error("mediapipe is required for the updated sign model.")
    st.code("mediapipe==0.10.14", language="text")
    st.caption(MEDIAPIPE_ERROR)
    st.stop()

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.subheader("Camera")
    ctx = webrtc_streamer(
        key="ishara-auto-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30, "max": 30},
                "facingMode": "user",
            },
            "audio": False,
        },
        video_frame_callback=_video_callback,
        async_processing=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Start Auto", type="primary", use_container_width=True):
            st.session_state["auto_live_running"] = True
            st.rerun()
    with b2:
        if st.button("Stop", use_container_width=True):
            st.session_state["auto_live_running"] = False
            st.rerun()
    with b3:
        if st.button("Undo", use_container_width=True):
            _undo_last()
            st.rerun()
    with b4:
        if st.button("Clear", use_container_width=True):
            _clear_sentence()
            st.rerun()

    if st.button("Speak Sentence", use_container_width=True):
        _speak_sentence()

    if st.session_state.get("last_audio_path"):
        st.audio(st.session_state["last_audio_path"])

    if st.session_state.get("last_error"):
        st.warning(st.session_state["last_error"])

with right:
    conf = st.session_state.get("current_confidence")
    conf_text = "---" if conf is None else f"{float(conf):.3f}"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="soft-muted">Current Prediction</div>
            <div class="big-live-word">{st.session_state.get("current_prediction", "---")}</div>
            <div class="soft-muted">Confidence: {conf_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sentence = " ".join(st.session_state.get("sentence_words", [])).strip() or "---"
    gloss_sequence = " + ".join(st.session_state.get("gloss_sequence", [])) or "---"
    st.markdown("### Live English Sentence")
    st.markdown(f"## {sentence}")
    st.markdown("### Gloss Sequence")
    st.write(gloss_sequence)
    with st.expander("Raw live result", expanded=False):
        st.json(st.session_state.get("last_raw", {}))

st.divider()
st.subheader("Speech to Text for hearing speaker")
st.caption("The hearing person can record or upload speech; Ishara converts it to text for the deaf user to read.")

audio_input = None
try:
    audio_input = st.audio_input("Record speech")
except Exception:
    audio_input = None
uploaded_audio = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "ogg", "webm"], key="live_stt_upload")
chosen_audio = audio_input or uploaded_audio
if st.button("Convert Speech to Text", use_container_width=True):
    _run_stt(chosen_audio)
if st.session_state.get("stt_text"):
    st.success("Speech converted to text")
    st.markdown(f"## {st.session_state['stt_text']}")

if st.session_state.get("auto_live_running"):
    if ctx and ctx.state.playing:
        _process_live_chunk(
            top_k=top_k,
            chunk_ms=chunk_ms,
            live_min_confidence=live_min_confidence,
            release_confirm_confidence=release_confirm_confidence,
            hold_confirm_confidence=hold_confirm_confidence,
            stable_count_required=stable_count_required,
            repeat_cooldown_ms=repeat_cooldown_ms,
        )
        time.sleep(max(0.05, loop_gap_ms / 1000.0))
        st.rerun()
    else:
        st.info("Open the camera first, then click Start Auto.")
