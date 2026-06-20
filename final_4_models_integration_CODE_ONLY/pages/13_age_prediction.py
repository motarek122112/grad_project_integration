from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.title("🎂 Age Prediction")
st.caption("Fast page load. The age model is warmed silently after login.")

source_mode = st.radio("Input", ["Upload image", "Camera"], horizontal=True)
image_file = None
if source_mode == "Upload image":
    image_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("Take face photo")


def _extract_age(result: Any):
    if isinstance(result, dict):
        for k in ["age", "predicted_age", "estimated_age", "prediction", "label"]:
            if k in result:
                return result.get(k)
    return result


def _run_age(image_path: Path) -> Dict[str, Any]:
    try:
        import model_adapters.age_prediction as age_adapter
        names = ["predict_age", "estimate_age", "run_age_prediction", "analyze_age", "predict_image", "predict", "run"]
        last_error = None
        for name in names:
            fn = getattr(age_adapter, name, None)
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
                    return {"ok": True, "age": _extract_age(raw), "raw_result": raw, "adapter_function": name}
                except TypeError as exc:
                    last_error = exc
                    continue
        return {"ok": False, "error": "No compatible age function found in model_adapters/age_prediction.py", "last_error": str(last_error) if last_error else None}
    except Exception as exc:
        import traceback
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)}

if image_file is not None:
    st.image(image_file, use_container_width=True)

if st.button("Predict Age", type="primary", use_container_width=True):
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
    with st.spinner("Predicting age..."):
        result = _run_age(image_path)
    if not result.get("ok"):
        st.error(result.get("error", "Age prediction failed."))
        with st.expander("Raw error", expanded=False):
            st.json(result)
        st.stop()
    st.success("Age predicted.")
    st.metric("Predicted age", result.get("age", "---"))
    with st.expander("Raw result", expanded=False):
        st.json(result)
