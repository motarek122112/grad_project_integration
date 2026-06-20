from __future__ import annotations

import importlib
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WARMUP_LOCK = threading.Lock()
_WARMUP_STARTED = False
_WARMUP_DONE = False
_WARMUP_ERROR: Optional[str] = None


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


def _make_silent_wav(path: Path, seconds: float = 0.35, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(seconds * sample_rate))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)
    return path


def _get_speech_adapter():
    return importlib.import_module("model_adapters.speech_to_text")


def _call_stt_function(fn, audio_path: Path) -> Dict[str, Any]:
    """Try Path first. The current adapter calls audio_path.exists(), so str causes AttributeError."""
    audio_path = Path(audio_path)
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

    last_error = None
    for attempt_name, attempt in attempts:
        try:
            raw = attempt()
            return {
                "ok": True,
                "text": _extract_text(raw),
                "raw_result": raw,
                "attempt": attempt_name,
            }
        except TypeError as exc:
            last_error = exc
            continue
        except AttributeError as exc:
            # Old bug: str has no .exists(). Continue to Path attempts / next signatures.
            last_error = exc
            continue

    return {
        "ok": False,
        "error": str(last_error) if last_error else "No compatible STT call signature worked.",
    }


def run_speech_to_text_fast(audio_path: str | Path) -> Dict[str, Any]:
    """Fast, robust STT runner for both the STT page and the Live page."""
    start = time.perf_counter()
    try:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return {"ok": False, "error": f"Audio file not found: {audio_path}"}

        speech_adapter = _get_speech_adapter()

        # Prefer the known adapter function first.
        names = [
            "transcribe_audio",
            "transcribe_speech",
            "speech_to_text",
            "predict_speech",
            "run_speech_to_text",
            "predict",
        ]

        last = None
        for name in names:
            fn = getattr(speech_adapter, name, None)
            if not callable(fn):
                continue
            result = _call_stt_function(fn, audio_path)
            if result.get("ok"):
                result["adapter_function"] = name
                result["time_ms"] = round((time.perf_counter() - start) * 1000, 2)
                return result
            last = result

        return last or {"ok": False, "error": "No compatible STT function found in model_adapters.speech_to_text.py"}

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "time_ms": round((time.perf_counter() - start) * 1000, 2),
        }


def warm_stt_once() -> Dict[str, Any]:
    """Warm the speech model using a very short silent WAV so the first real recording is faster."""
    global _WARMUP_DONE, _WARMUP_ERROR

    if _WARMUP_DONE:
        return {"ok": True, "status": "already-warmed"}

    with _WARMUP_LOCK:
        if _WARMUP_DONE:
            return {"ok": True, "status": "already-warmed"}

        try:
            # Import adapter and call optional loader functions if present.
            speech_adapter = _get_speech_adapter()
            for name in [
                "load_speech_model",
                "load_model",
                "load_stt_model",
                "load_speech_backend",
                "load_stt_backend",
                "get_model",
                "get_backend",
            ]:
                fn = getattr(speech_adapter, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass

            warm_dir = PROJECT_ROOT / "outputs" / "_warmup"
            warm_path = _make_silent_wav(warm_dir / "stt_silent_warmup.wav")
            result = run_speech_to_text_fast(warm_path)

            # Even if silent transcription returns no text, the heavy model path has been loaded.
            _WARMUP_DONE = True
            _WARMUP_ERROR = None if result.get("ok") else result.get("error")
            return {"ok": True, "status": "warmed", "stt_result": result}

        except Exception as exc:
            _WARMUP_ERROR = str(exc)
            return {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=6)}


def _warm_worker() -> None:
    warm_stt_once()


def start_stt_warmup() -> None:
    """Start STT warm-up in a background thread once per process."""
    global _WARMUP_STARTED
    if _WARMUP_STARTED or _WARMUP_DONE:
        return
    _WARMUP_STARTED = True
    thread = threading.Thread(target=_warm_worker, daemon=True, name="ishara-stt-warmup")
    thread.start()


def get_stt_warmup_status() -> Dict[str, Any]:
    return {
        "started": _WARMUP_STARTED,
        "done": _WARMUP_DONE,
        "error": _WARMUP_ERROR,
    }
