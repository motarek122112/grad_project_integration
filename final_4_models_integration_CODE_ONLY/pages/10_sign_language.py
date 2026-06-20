from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_adapters.sign_language import predict_sign_language

st.title("🤟 Sign Language to Text")
st.caption("Updated Landmark Transformer sign model integration")

st.info(
    "Use this page for Mohamed's updated Sign Language model only. "
    "The other models remain connected through their own adapters."
)

mode_label = st.radio(
    "Prediction mode",
    ["Single isolated sign", "Multi-sign sentence"],
    horizontal=True,
)

uploaded_video = st.file_uploader(
    "Upload a sign video",
    type=["mp4", "webm", "avi", "mov", "mkv"],
)

with st.expander("Advanced settings", expanded=False):
    top_k = st.slider("Top-K predictions", 1, 10, 5)
    threshold = st.slider("Segmentation threshold", 0.01, 0.30, 0.08, 0.01)
    min_pause_sec = st.slider("Minimum pause between signs", 0.10, 1.00, 0.35, 0.05)
    min_segment_sec = st.slider("Minimum segment length", 0.10, 1.00, 0.30, 0.05)
    confidence_threshold = st.slider("Sentence confidence threshold", 0.01, 0.50, 0.08, 0.01)
    use_language_decoder = st.checkbox("Use sentence language decoder", value=True)

if uploaded_video is not None:
    st.video(uploaded_video)

run = st.button("Run Sign Model", type="primary", use_container_width=True)

if run:
    if uploaded_video is None:
        st.warning("Please upload a video first.")
        st.stop()

    suffix = Path(uploaded_video.name).suffix or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_video.getbuffer())
        video_path = Path(tmp.name)

    sentence_mode = mode_label == "Multi-sign sentence"

    with st.spinner("Running updated sign model..."):
        result = predict_sign_language(
            video_path=video_path,
            sentence=sentence_mode,
            top_k=top_k,
            threshold=threshold,
            min_pause_sec=min_pause_sec,
            min_segment_sec=min_segment_sec,
            confidence_threshold=confidence_threshold,
            use_language_decoder=use_language_decoder,
        )

    if not result.get("ok"):
        st.error(result.get("error", "Sign model failed."))
        with st.expander("Traceback", expanded=False):
            st.code(result.get("traceback", ""))
        st.stop()

    text = result.get("sentence") or result.get("text") or result.get("gloss") or ""

    st.success("Prediction completed")

    st.subheader("Output")
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:16px;background:#f8fafc;border:1px solid #e5e7eb;">
            <div style="font-size:14px;color:#64748b;margin-bottom:6px;">Detected text</div>
            <div style="font-size:34px;font-weight:800;color:#0f766e;">{text or '---'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get("mode") == "single":
        c1, c2 = st.columns(2)
        c1.metric("Gloss", result.get("gloss") or "---")
        c2.metric("Confidence", f"{float(result.get('confidence') or 0):.3f}")

        if result.get("top_k"):
            st.subheader("Top-K Predictions")
            st.dataframe(result.get("top_k"), use_container_width=True)

    else:
        if result.get("gloss_sequence"):
            st.subheader("Gloss Sequence")
            st.write(" + ".join([str(x) for x in result.get("gloss_sequence")]))

        if result.get("word_sequence"):
            st.subheader("Word Sequence")
            st.write(" ".join([str(x) for x in result.get("word_sequence")]))

        st.caption(f"Language decoder used: {result.get('lm_decoder_used', False)}")

    with st.expander("Raw result", expanded=False):
        st.json(result)

    try:
        video_path.unlink(missing_ok=True)
    except Exception:
        pass
