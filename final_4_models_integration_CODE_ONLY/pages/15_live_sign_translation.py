from __future__ import annotations

import os
import sys
import time
import json
import html
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Low-latency imports
# -----------------------------------------------------------------------------
try:
    import av  # type: ignore
    import numpy as np  # type: ignore
    import cv2  # type: ignore
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    WEBRTC_OK = True
    WEBRTC_ERROR = None
except Exception as exc:  # pragma: no cover
    WEBRTC_OK = False
    WEBRTC_ERROR = exc

try:
    from model_adapters.sign_language import predict_single_sign
except Exception:
    predict_single_sign = None

try:
    from model_adapters.text_to_speech import synthesize_speech
except Exception:
    synthesize_speech = None

try:
    from model_adapters.speech_to_text import transcribe_audio
except Exception:
    transcribe_audio = None

# -----------------------------------------------------------------------------
# Page CSS: keep original dark app style, but make camera cleaner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .ishara-live-title {font-size: 44px; font-weight: 900; margin-bottom: 0.2rem;}
    .ishara-muted {opacity: .72; font-size: 15px; margin-bottom: 1.0rem;}
    .ishara-card {
        border: 1px solid rgba(148,163,184,.28);
        border-radius: 18px;
        padding: 18px;
        background: rgba(15,23,42,.40);
        margin-bottom: 16px;
    }
    .ishara-output {
        border-radius: 18px;
        padding: 22px;
        background: rgba(30,41,59,.78);
        border: 1px solid rgba(148,163,184,.24);
        font-size: 30px;
        font-weight: 800;
        min-height: 92px;
    }
    .ishara-status {
        border-radius: 14px;
        padding: 13px 16px;
        background: rgba(234,179,8,.18);
        color: #fde68a;
        border: 1px solid rgba(234,179,8,.22);
        font-weight: 700;
    }
    .ishara-good {
        border-radius: 14px;
        padding: 13px 16px;
        background: rgba(34,197,94,.13);
        color: #bbf7d0;
        border: 1px solid rgba(34,197,94,.20);
        font-weight: 700;
    }
    div[data-testid="stHorizontalBlock"] button {min-height: 54px; font-weight: 800;}
    /* Try to improve the WebRTC video display inside Streamlit */
    video {border-radius: 18px !important; object-fit: cover !important; width: 100% !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, default in {
    "live_auto_enabled": False,
    "live_words": [],
    "live_glosses": [],
    "live_last_word": "",
    "live_last_accept_ts": 0.0,
    "live_current_text": "---",
    "live_current_conf": 0.0,
    "live_status": "Open camera, then press Start Auto.",
    "live_last_raw": {},
}.items():
    st.session_state.setdefault(key, default)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _user_profile() -> Dict[str, Any]:
    return st.session_state.get("user", {}) or {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw or {}
    top_k = raw.get("top_k") or raw.get("top_predictions") or []
    top1 = top_k[0] if isinstance(top_k, list) and top_k else {}
    text = (
        raw.get("text")
        or raw.get("word")
        or raw.get("clean_text")
        or raw.get("predicted_text")
        or top1.get("text")
        or top1.get("word")
        or raw.get("gloss")
        or top1.get("gloss")
        or raw.get("label")
        or top1.get("label")
        or ""
    )
    gloss = raw.get("gloss") or top1.get("gloss") or raw.get("label") or top1.get("label") or text
    conf = (
        raw.get("confidence")
        or raw.get("probability")
        or raw.get("score")
        or top1.get("confidence")
        or top1.get("probability")
        or top1.get("score")
        or 0.0
    )
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    return {"text": _safe_text(text), "gloss": _safe_text(gloss), "confidence": conf, "raw": raw}


def _write_frames_to_video(frames: List[Any], fps: int = 12) -> Path:
    """Write BGR frames to a temporary MP4 file for the existing sign adapter."""
    if not frames:
        raise ValueError("No frames available.")
    first = frames[0]
    h, w = first.shape[:2]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    path = Path(tmp.name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        writer.release()
    return path


def _predict_live_chunk(frames: List[Any], top_k: int) -> Dict[str, Any]:
    if predict_single_sign is None:
        return {"ok": False, "error": "Sign model adapter is not available."}
    video_path = None
    try:
        video_path = _write_frames_to_video(frames, fps=12)
        result = predict_single_sign(video_path=video_path, top_k=top_k)
        return result or {"ok": False, "error": "Empty sign model result."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        try:
            if video_path:
                Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


class LowLatencyVideoProcessor(VideoProcessorBase):
    """Stores the latest camera frames without running inference in recv().

    This keeps camera preview smoother. Prediction happens only every tick from the
    Streamlit page using the latest stored frames.
    """

    def __init__(self) -> None:
        self.frames = deque(maxlen=28)
        self.last_frame_ts = 0.0
        self.total_frames = 0

    def recv(self, frame):  # type: ignore[override]
        img = frame.to_ndarray(format="bgr24")
        # Downscale only for model chunk memory/speed; preview remains WebRTC-native.
        h, w = img.shape[:2]
        if w > 640:
            scale = 640.0 / float(w)
            img_small = cv2.resize(img, (640, max(1, int(h * scale))))
        else:
            img_small = img
        self.frames.append(img_small)
        self.total_frames += 1
        self.last_frame_ts = time.time()
        return frame

    def get_recent_frames(self, max_frames: int = 14) -> List[Any]:
        frames = list(self.frames)
        if not frames:
            return []
        return frames[-max_frames:]


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.markdown('<div class="ishara-live-title">🎥 Live Sign Translation</div>', unsafe_allow_html=True)
st.markdown('<div class="ishara-muted">Auto live mode inside Streamlit — no separate FastAPI server.</div>', unsafe_allow_html=True)

with st.expander("Live settings", expanded=False):
    preset = st.selectbox(
        "Camera mode",
        ["Smooth / low latency", "Balanced", "Higher quality"],
        index=0,
        help="Smooth is recommended for Streamlit Cloud. Higher quality may lag on Cloud CPU.",
    )
    top_k = st.slider("Top-K", 1, 10, 5)
    min_conf = st.slider("Minimum confidence", 0.05, 0.95, 0.30, 0.01)
    hold_conf = st.slider("Hold confirm confidence", 0.10, 0.95, 0.62, 0.01)
    release_conf = st.slider("Release confirm confidence", 0.01, 0.50, 0.15, 0.01)
    repeat_cooldown_ms = st.slider("Repeat cooldown ms", 500, 5000, 2500, 100)
    min_frames = st.slider("Frames per prediction", 4, 20, 6, 1)

if not WEBRTC_OK:
    st.error(f"streamlit-webrtc / av is not available: {WEBRTC_ERROR}")
    st.code("streamlit-webrtc>=0.72,<1\nav>=12\nmediapipe==0.10.14", language="text")
    st.stop()

# Camera constraints: do not push 1280x720 through Streamlit Cloud by default.
if preset == "Higher quality":
    width, height, fps = 960, 540, 20
elif preset == "Balanced":
    width, height, fps = 640, 480, 20
else:
    width, height, fps = 480, 360, 15

rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]
}

ctx = webrtc_streamer(
    key="ishara-online-auto-live-low-latency-v3",
    video_processor_factory=LowLatencyVideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {
            "width": {"ideal": width},
            "height": {"ideal": height},
            "frameRate": {"ideal": fps, "max": fps},
            "facingMode": "user",
        },
        "audio": False,
    },
    async_processing=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Start Auto", type="primary", use_container_width=True):
        st.session_state.live_auto_enabled = True
        st.session_state.live_status = "Auto live started. Hold your sign in the center."
        st.rerun()
with c2:
    if st.button("Stop", use_container_width=True):
        st.session_state.live_auto_enabled = False
        st.session_state.live_status = "Auto live stopped."
        st.rerun()
with c3:
    if st.button("Undo", use_container_width=True):
        if st.session_state.live_words:
            st.session_state.live_words.pop()
        if st.session_state.live_glosses:
            st.session_state.live_glosses.pop()
        st.rerun()
with c4:
    if st.button("Clear", use_container_width=True):
        st.session_state.live_words = []
        st.session_state.live_glosses = []
        st.session_state.live_current_text = "---"
        st.session_state.live_current_conf = 0.0
        st.session_state.live_last_word = ""
        st.rerun()

sentence = " ".join(st.session_state.live_words).strip()


def _render_browser_speak_sentence_button(text_to_speak: str) -> None:
    """Speak instantly in the user's browser.

    This avoids slow server-side TTS on Streamlit Cloud and works as a real
    user-click action, so Chrome/Edge do not block the audio.
    """

    safe_text = str(text_to_speak or "").strip()
    js_text = json.dumps(safe_text)
    disabled = "disabled" if not safe_text else ""
    opacity = "0.45" if not safe_text else "1"
    label = "Speak Sentence" if safe_text else "Speak Sentence"

    components.html(
        f"""
        <div style="width:100%;">
          <button id="isharaSpeakBtn" {disabled}
            style="
              width:100%; min-height:54px; border-radius:14px;
              border:1px solid rgba(148,163,184,.35);
              background:rgba(15,23,42,.55); color:#ffffff;
              font-weight:800; font-size:16px; cursor:pointer; opacity:{opacity};
            ">
            {html.escape(label)}
          </button>
          <div id="isharaSpeakStatus" style="font-family:Arial,sans-serif;font-size:12px;opacity:.72;margin-top:6px;color:#cbd5e1;"></div>
        </div>
        <script>
          const btn = document.getElementById("isharaSpeakBtn");
          const statusEl = document.getElementById("isharaSpeakStatus");
          const text = {js_text};

          function pickVoice() {{
            const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
            return voices.find(v => (v.lang || '').toLowerCase().startsWith('en') && /female|zira|samantha|google/i.test(v.name))
                || voices.find(v => (v.lang || '').toLowerCase().startsWith('en'))
                || voices[0]
                || null;
          }}

          function speakNow() {{
            if (!text || !text.trim()) {{
              if (statusEl) statusEl.textContent = "No sentence to speak yet.";
              return;
            }}
            if (!('speechSynthesis' in window)) {{
              if (statusEl) statusEl.textContent = "Browser speech is not supported here.";
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            const voice = pickVoice();
            if (voice) utterance.voice = voice;
            utterance.onstart = () => {{ if (statusEl) statusEl.textContent = "Speaking..."; }};
            utterance.onend = () => {{ if (statusEl) statusEl.textContent = ""; }};
            utterance.onerror = () => {{ if (statusEl) statusEl.textContent = "Speech failed. Try clicking again."; }};
            window.speechSynthesis.speak(utterance);
          }}

          if (btn) {{
            btn.onclick = speakNow;
          }}
          if (window.speechSynthesis) {{
            window.speechSynthesis.onvoiceschanged = () => pickVoice();
          }}
        </script>
        """,
        height=82,
    )


_render_browser_speak_sentence_button(sentence)

left, right = st.columns([1.25, 1])
with left:
    st.markdown("### Current Prediction")
    conf = float(st.session_state.live_current_conf or 0.0)
    current = st.session_state.live_current_text or "---"
    st.markdown(f'<div class="ishara-output">{current}<br><span style="font-size:15px;opacity:.7;">Confidence: {conf:.3f}</span></div>', unsafe_allow_html=True)
    if st.session_state.live_status:
        klass = "ishara-good" if "Accepted" in st.session_state.live_status else "ishara-status"
        st.markdown(f'<div class="{klass}">{st.session_state.live_status}</div>', unsafe_allow_html=True)

with right:
    st.markdown("### Live Sentence")
    st.markdown(f'<div class="ishara-output" style="font-size:24px;">{sentence or "---"}</div>', unsafe_allow_html=True)
    if st.session_state.live_glosses:
        st.caption("Gloss: " + " + ".join(st.session_state.live_glosses))

# -----------------------------------------------------------------------------
# Auto prediction tick. Keep this after UI so page renders before inference.
# -----------------------------------------------------------------------------
processor = ctx.video_processor if ctx and ctx.state.playing else None
if st.session_state.live_auto_enabled:
    if processor is None:
        st.session_state.live_status = "Waiting for camera to start..."
        time.sleep(0.35)
        st.rerun()
    else:
        frames = processor.get_recent_frames(max_frames=max(8, min_frames + 2))
        if len(frames) < min_frames:
            st.session_state.live_status = f"Waiting for camera frames... {len(frames)}/{min_frames}"
            time.sleep(0.45)
            st.rerun()
        else:
            t0 = time.time()
            raw = _predict_live_chunk(frames[-min_frames:], top_k=top_k)
            elapsed_ms = int((time.time() - t0) * 1000)
            st.session_state.live_last_raw = raw

            if not raw.get("ok"):
                err = raw.get("error", "Prediction failed.")
                st.session_state.live_status = f"Prediction error: {err}"
                time.sleep(0.8)
                st.rerun()

            pred = _normalize_result(raw)
            word = pred["text"]
            gloss = pred["gloss"]
            confidence = float(pred["confidence"] or 0.0)
            now = time.time()
            cooldown_ok = (now - float(st.session_state.live_last_accept_ts or 0.0)) * 1000.0 >= repeat_cooldown_ms
            duplicate = word and word.lower() == str(st.session_state.live_last_word or "").lower()

            st.session_state.live_current_text = word or "---"
            st.session_state.live_current_conf = confidence

            if not word:
                st.session_state.live_status = f"No clear sign yet. {elapsed_ms} ms"
            elif confidence >= hold_conf and (not duplicate or cooldown_ok):
                st.session_state.live_words.append(word)
                if gloss:
                    st.session_state.live_glosses.append(gloss)
                st.session_state.live_last_word = word
                st.session_state.live_last_accept_ts = now
                st.session_state.live_status = f"Accepted: {word} ({confidence:.3f}) • {elapsed_ms} ms"
            elif confidence >= min_conf:
                st.session_state.live_status = f"Candidate: {word} ({confidence:.3f}) • hold sign steady"
            elif confidence <= release_conf:
                st.session_state.live_status = f"Release / no stable sign ({confidence:.3f})"
            else:
                st.session_state.live_status = f"Low confidence: {word} ({confidence:.3f})"

            # Do not hammer Streamlit Cloud. Local mode can reduce this to 0.15.
            time.sleep(0.25)
            st.rerun()

with st.expander("Speech to Text for hearing speaker", expanded=False):
    st.caption("The hearing person can record/upload audio, then the deaf user reads the transcription.")
    audio_file = None
    if hasattr(st, "audio_input"):
        audio_file = st.audio_input("Record voice")
    if audio_file is None:
        audio_file = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "ogg", "webm"])

    if audio_file is not None and st.button("Convert Speech to Text", use_container_width=True):
        if transcribe_audio is None:
            st.error("Speech-to-text adapter is not available.")
        else:
            suffix = Path(getattr(audio_file, "name", "audio.wav")).suffix or ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(audio_file.getbuffer())
            tmp.close()
            audio_path = Path(tmp.name)
            try:
                with st.spinner("Transcribing..."):
                    result = transcribe_audio(audio_path=audio_path)
                text = result.get("text") or result.get("transcription") or result.get("sentence") or ""
                st.markdown(f'<div class="ishara-output" style="font-size:24px;">{text or "---"}</div>', unsafe_allow_html=True)
                with st.expander("Raw STT result", expanded=False):
                    st.json(result)
            except Exception as exc:
                st.error(f"STT failed: {exc}")
            finally:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception:
                    pass

with st.expander("Raw live result", expanded=False):
    st.json(st.session_state.live_last_raw)
