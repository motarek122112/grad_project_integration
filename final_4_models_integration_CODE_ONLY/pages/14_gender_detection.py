from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.title("🧑‍🤝‍🧑 Gender Detection")
st.caption("Fast page load. The gender model is warmed silently after login.")

source_mode = st.radio("Input", ["Upload image", "Camera"], horizontal=True)
image_file = None
if source_mode == "Upload image":
    image_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("Take face photo")


def _extract_gender(result: Any):
    if isinstance(result, dict):
        for k in ["gender", "predicted_gender", "label", "prediction", "class"]:
            if k in result:
                return result.get(k)
    return result


def _extract_confidence(result: Any):
    if isinstance(result, dict):
        for k in ["confidence", "probability", "score"]:
            if k in result:
                return result.get(k)
    return None


def _run_gender(image_path: Path) -> Dict[str, Any]:
    try:
        import model_adapters.gender_detection as gender_adapter
        names = ["predict_gender", "detect_gender", "run_gender_detection", "analyze_gender", "predict_image", "predict", "run"]
        last_error = None
        for name in names:
            fn = getattr(gender_adapter, name, None)
            if not callable(fn):
                continue
            attempts = [
                lambda: fn(image_path=str(image_path)),
                lambda: fn(image_file=str(image_path)),
                lambda: fn(file_path=str(image_path)),
                lambda: fn(path=str(image_path)),
                lambda: fn(str(image_path)),
            ]
            for attempt in attempts:
                try:
                    raw = attempt()
                    return {"ok": True, "gender": _extract_gender(raw), "confidence": _extract_confidence(raw), "raw_result": raw, "adapter_function": name}
                except TypeError as exc:
                    last_error = exc
                    continue
        return {"ok": False, "error": "No compatible gender function found in model_adapters/gender_detection.py", "last_error": str(last_error) if last_error else None}
    except Exception as exc:
        import traceback
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)}

if image_file is not None:
    st.image(image_file, use_container_width=True)

if st.button("Detect Gender", type="primary", use_container_width=True):
    if image_file is None:
        st.warning("Upload or take an image first.")
        st.stop()
    suffix = ".jpg"
    name = getattr(image_file, "name", "face.jpg") or "face.jpg"
    if "." in name:
        suffix = "." + name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_file.getvalue())
        image_path = Path(tmp.name)
    with st.spinner("Detecting gender..."):
        result = _run_gender(image_path)
    if not result.get("ok"):
        st.error(result.get("error", "Gender detection failed."))
        with st.expander("Raw error", expanded=False):
            st.json(result)
        st.stop()
    st.success("Gender detected.")
    c1, c2 = st.columns(2)
    c1.metric("Gender", result.get("gender", "---"))
    conf = result.get("confidence")
    try:
        c2.metric("Confidence", f"{float(conf):.3f}" if conf is not None else "---")
    except Exception:
        c2.metric("Confidence", str(conf))
    with st.expander("Raw result", expanded=False):
        st.json(result)
