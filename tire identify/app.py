import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

from predictor import preprocess_tire_print, predict_car_model

# Page config
st.set_page_config(
    page_title="TirePrint → Car Model",
    page_icon="🚗",
    layout="wide",
)

# Minimal CSS for a modern look
st.markdown(
    """
    <style>
    .app-title { font-size: 2.2rem; font-weight: 700; }
    .subtle { color: #6b7280; }
    .card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border-radius: 14px; padding: 18px; color: #e5e7eb; box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .accent { color: #93c5fd; }
    .badge {
        display: inline-block; background: #111827; border: 1px solid #374151; color: #d1d5db; padding: 6px 10px; border-radius: 10px; font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Display Options")
    layout_mode = st.radio(
        "Layout style",
        ["Vertical Cards", "Split Horizontal"],
        index=0,
    )
    st.write("""Choose how the preview and results are arranged.""")

st.markdown('<div class="app-title">🔍 Tire Print Identifier</div>', unsafe_allow_html=True)
st.markdown("<div class='subtle'>Upload a tire print image to predict the car model.</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload tire print image", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=False)

if uploaded:
    # Load and normalize the image
    image_bytes = uploaded.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)  # respectful of camera orientation

    # Preprocess (edges/contrast) for an informative preview
    processed_preview = preprocess_tire_print(image)

    # Predict (mock/deterministic without a trained model)
    predicted_model, confidence = predict_car_model(image)

    if layout_mode == "Vertical Cards":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='badge'>Input</div>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Tire Print", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top: 16px;'>", unsafe_allow_html=True)
        st.markdown("<div class='badge'>Preprocessing</div>", unsafe_allow_html=True)
        st.image(processed_preview, caption="Edge/contrast preview", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top: 16px;'>", unsafe_allow_html=True)
        st.markdown("<div class='badge'>Prediction</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"<span class='accent' style='font-size:1.4rem'>Model: {predicted_model}</span>", unsafe_allow_html=True)
            st.write("Confidence")
            st.progress(int(confidence * 100))
        with col2:
            st.metric(label="Score", value=f"{confidence:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='badge'>Input</div>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Tire Print", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card' style='margin-top: 16px;'>", unsafe_allow_html=True)
            st.markdown("<div class='badge'>Preprocessing</div>", unsafe_allow_html=True)
            st.image(processed_preview, caption="Edge/contrast preview", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='badge'>Prediction</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='accent' style='font-size:1.4rem'>Model: {predicted_model}</span>", unsafe_allow_html=True)
            st.write("Confidence")
            st.progress(int(confidence * 100))
            st.metric(label="Score", value=f"{confidence:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("Try another photo"):
        st.experimental_rerun()
else:
    st.info("Upload a tire print image to begin.")
