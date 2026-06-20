from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.title("🎙️ Speech to Text")
st.caption("Convert spoken audio into readable text. The STT model is warmed silently after login.")

try:
    recorded_audio = st.audio_input("Record audio")
except Exception:
    recorded_audio = None
    st.info("Audio recorder is unavailable in this Streamlit version. Upload audio below.")

uploaded_audio = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(
            result.get("text")
            or result.get("transcript")
            or result.get("clean_text")
            or result.get("prediction")
            or result.get("sentence")
            or ""
        )
    return str(result or "")


def _run_stt(audio_path: Path) -> Dict[str, Any]:
    """
    Robust STT runner.
    Important fix: try pathlib.Path first because the current adapter uses audio_path.exists().
    """
    try:
        import model_adapters.speech_to_text as speech_adapter

        names = [
            "transcribe_audio",
            "transcribe_speech",
            "speech_to_text",
            "predict_speech",
            "run_speech_to_text",
            "predict",
        ]

        last_error = None

        for name in names:
            fn = getattr(speech_adapter, name, None)
            if not callable(fn):
                continue

            attempts = [
                ("audio_path_Path", lambda: fn(audio_path=audio_path)),
                ("audio_file_Path", lambda: fn(audio_file=audio_path)),
                ("file_path_Path", lambda: fn(file_path=audio_path)),
                ("path_Path", lambda: fn(path=audio_path)),
                ("positional_Path", lambda: fn(audio_path)),
                ("audio_path_str", lambda: fn(audio_path=str(audio_path))),
                ("audio_file_str", lambda: fn(audio_file=str(audio_path))),
                ("file_path_str", lambda: fn(file_path=str(audio_path))),
                ("path_str", lambda: fn(path=str(audio_path))),
                ("positional_str", lambda: fn(str(audio_path))),
            ]

            for attempt_name, attempt in attempts:
                try:
                    raw = attempt()
                    return {
                        "ok": True,
                        "text": _extract_text(raw),
                        "raw_result": raw,
                        "adapter_function": name,
                        "attempt": attempt_name,
                    }
                except TypeError as exc:
                    last_error = exc
                    continue
                except AttributeError as exc:
                    # Handles old bug: passing str into adapter that expects Path and calls .exists().
                    last_error = exc
                    continue

        return {
            "ok": False,
            "error": "No compatible STT function found in model_adapters/speech_to_text.py",
            "last_error": str(last_error) if last_error else None,
        }

    except Exception as exc:
        import traceback

        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


if st.button("Convert Speech to Text", type="primary", use_container_width=True):
    source = recorded_audio or uploaded_audio
    if source is None:
        st.warning("Record or upload audio first.")
        st.stop()

    name = getattr(source, "name", "recording.wav") or "recording.wav"
    suffix = "." + name.split(".")[-1].lower() if "." in name else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(source.getvalue())
        audio_path = Path(tmp.name)

    st.audio(str(audio_path))

    with st.spinner("Transcribing..."):
        result = _run_stt(audio_path)

    if not result.get("ok"):
        st.error(result.get("error", "Speech-to-text failed."))
        with st.expander("Raw error", expanded=False):
            st.json(result)
        st.stop()

    st.success("Done.")
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:16px;background:#f8fafc;border:1px solid #e5e7eb;">
            <div style="font-size:14px;color:#64748b;margin-bottom:8px;">Transcript</div>
            <div style="font-size:32px;font-weight:800;color:#0f766e;line-height:1.35;">{result.get('text') or '---'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Raw result", expanded=False):
        st.json(result)
