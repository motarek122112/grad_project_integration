from __future__ import annotations

import importlib
import json
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = PROJECT_ROOT / "data" / "preload_status.json"
STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_status(status: dict) -> None:
    try:
        status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        return exc


def _try_call(fn: Callable[..., Any], attempts: Iterable[Callable[[], Any]]) -> bool:
    for attempt in attempts:
        try:
            attempt()
            return True
        except TypeError:
            continue
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False


def _warm_tts() -> str:
    mod = _try_import("tts_inference")
    if isinstance(mod, Exception):
        return f"skip: {mod}"

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "_warmup_tts.wav"

    try:
        if hasattr(mod, "get_pipeline"):
            mod.get_pipeline("a")
    except Exception:
        pass

    fn = getattr(mod, "text_to_speech", None)
    if not callable(fn):
        return "skip: text_to_speech not found"

    attempts = [
        lambda: fn("Hello", str(out_path), gender="Male", age=34),
        lambda: fn("Hello", output_path=str(out_path), gender="Male", age=34),
        lambda: fn("Hello", output_path=str(out_path), gender="Male"),
        lambda: fn("Hello", str(out_path)),
    ]
    return "ok" if _try_call(fn, attempts) else "partial"


def _warm_stt() -> str:
    try:
        from app.live_stt_fast import warm_stt_once
        result = warm_stt_once()
        return "ok" if result.get("ok") else f"partial: {result.get('error')}"
    except Exception as exc:
        return f"skip: {exc}"

def _warm_image_module(module_name: str, sample_name: str) -> str:
    sample_candidates = [
        PROJECT_ROOT / "samples" / sample_name,
        PROJECT_ROOT / "samples" / "sample_person.jpg",
        PROJECT_ROOT / "samples" / "sample_person_2.jpg",
    ]
    sample = next((p for p in sample_candidates if p.exists()), None)
    mod = _try_import(module_name)
    if isinstance(mod, Exception):
        return f"skip: {mod}"

    for name in ["load_model", "load_age_model", "load_gender_model", "load_backend", "get_model", "get_age_model", "get_gender_model"]:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    if sample is None:
        return "loaded-no-sample"

    names = [
        "predict_age", "estimate_age", "run_age_prediction", "analyze_age",
        "predict_gender", "detect_gender", "run_gender_detection", "analyze_gender",
        "predict_image", "predict", "run",
    ]
    for name in names:
        fn = getattr(mod, name, None)
        if not callable(fn):
            continue
        attempts = [
            lambda fn=fn: fn(image_path=str(sample)),
            lambda fn=fn: fn(image_file=str(sample)),
            lambda fn=fn: fn(file_path=str(sample)),
            lambda fn=fn: fn(path=str(sample)),
            lambda fn=fn: fn(str(sample)),
        ]
        if _try_call(fn, attempts):
            return "ok"
    return "loaded"


def _warm_sign() -> str:
    mod = _try_import("model_adapters.sign_language")
    if isinstance(mod, Exception):
        return f"skip: {mod}"
    for name in ["load_updated_sign_model", "get_sign_model", "load_sign_model", "load_model"]:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                fn()
                return "ok"
            except Exception:
                pass
    return "loaded"


def _preload_worker() -> None:
    status = {"running": True, "done": False, "items": {}}
    _write_status(status)
    items = [
        ("Sign Language", _warm_sign),
        ("Text to Speech", _warm_tts),
        ("Speech to Text", _warm_stt),
        ("Age Prediction", lambda: _warm_image_module("model_adapters.age_prediction", "sample_person.jpg")),
        ("Gender Detection", lambda: _warm_image_module("model_adapters.gender_detection", "sample_person.jpg")),
    ]
    for label, fn in items:
        try:
            status["items"][label] = fn()
        except Exception:
            status["items"][label] = traceback.format_exc(limit=3)
        _write_status(status)
    status["running"] = False
    status["done"] = True
    _write_status(status)


@st.cache_resource(show_spinner=False)
def _start_preload_thread_once():
    thread = threading.Thread(target=_preload_worker, daemon=True, name="ishara-fast-preload")
    thread.start()
    return thread


def start_fast_preload() -> None:
    """Start silent background warm-up once per Streamlit process."""
    try:
        _start_preload_thread_once()
    except Exception:
        pass


def get_preload_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"running": False, "done": False, "items": {}}
