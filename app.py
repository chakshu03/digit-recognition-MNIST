from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.data_utils import prepare_uploaded_image


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "mnist_cnn_model.keras"
METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
HISTORY_CSV_PATH = ARTIFACTS_DIR / "training_history.csv"
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / "classification_report.csv"
HISTORY_PLOT_PATH = ARTIFACTS_DIR / "history_plot.png"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
MODEL_SUMMARY_PATH = ARTIFACTS_DIR / "model_summary.txt"
MISCLASSIFIED_PLOT_PATH = ARTIFACTS_DIR / "misclassified_samples.png"
FEATURE_MAPS_PATH = ARTIFACTS_DIR / "feature_maps.png"
SAMPLE_PREDICTION_PATH = ARTIFACTS_DIR / "sample_prediction.png"


st.set_page_config(
    page_title="DigitScope CNN",
    page_icon="🔢",
    layout="wide",
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_metadata() -> dict:
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 221, 153, 0.28), transparent 30%),
                radial-gradient(circle at top right, rgba(73, 146, 255, 0.18), transparent 28%),
                linear-gradient(180deg, #fffaf1 0%, #f4f7fb 52%, #eef3f8 100%);
        }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .hero-card, .glass-card, .metric-card {
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        }
        .hero-card { padding: 2rem; margin-bottom: 1rem; }
        .glass-card { padding: 1.2rem 1.2rem 0.8rem 1.2rem; margin-bottom: 1rem; }
        .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.9rem; margin: 1rem 0 1.3rem 0; }
        .metric-card { padding: 1rem; }
        .metric-label { font-size: 0.82rem; letter-spacing: 0.06em; text-transform: uppercase; color: #64748b; margin-bottom: 0.35rem; }
        .metric-value { font-size: 1.6rem; font-weight: 700; color: #0f172a; line-height: 1.1; }
        .metric-note { margin-top: 0.25rem; color: #475569; font-size: 0.92rem; }
        .hero-kicker { color: #b45309; font-size: 0.92rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        .hero-title { font-size: 3rem; line-height: 1.02; color: #0f172a; margin: 0.2rem 0 0.8rem 0; font-weight: 800; }
        .hero-subtitle { color: #334155; font-size: 1.05rem; max-width: 60rem; }
        .pill-row { display: flex; flex-wrap: wrap; gap: 0.65rem; margin-top: 1rem; }
        .pill { padding: 0.45rem 0.8rem; border-radius: 999px; background: #fff4d8; color: #92400e; font-weight: 600; font-size: 0.9rem; border: 1px solid rgba(146, 64, 14, 0.12); }
        .section-title { font-size: 1.2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.75rem; }
        .section-copy { color: #475569; margin-bottom: 0.85rem; }
        .prediction-banner { border-radius: 20px; padding: 1rem 1.1rem; background: linear-gradient(135deg, #0f766e, #155e75); color: white; margin-bottom: 1rem; }
        .prediction-digit { font-size: 2.8rem; font-weight: 800; margin: 0.15rem 0; }
        .small-muted { color: #64748b; font-size: 0.92rem; }
        @media (max-width: 980px) {
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hero-title { font-size: 2.2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metadata: dict) -> None:
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">{metadata['test_accuracy']:.2%}</div>
                <div class="metric-note">CNN on MNIST test set</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test Loss</div>
                <div class="metric-value">{metadata['test_loss']:.4f}</div>
                <div class="metric-note">Categorical cross-entropy</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Parameters</div>
                <div class="metric-value">{metadata['total_parameters']:,}</div>
                <div class="metric-note">Trainable CNN weights</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Misclassifications</div>
                <div class="metric-value">{metadata['misclassified_samples']}</div>
                <div class="metric-note">Out of 10,000 test images</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metadata: dict) -> None:
    architecture = " • ".join(metadata["architecture"])
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">CNN Digit Recognition</div>
            <div class="hero-title">DigitScope for MNIST</div>
            <div class="hero-subtitle">
                Upload a handwritten digit image and inspect how the trained convolutional network responds.
                This app pairs live prediction with experiment artifacts from the notebook-style build.
            </div>
            <div class="pill-row">
                <div class="pill">Dataset: {metadata['dataset']}</div>
                <div class="pill">Epochs: {metadata['epochs']}</div>
                <div class="pill">Batch Size: {metadata['batch_size']}</div>
                <div class="pill">Architecture: {architecture}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_panel(model, metadata: dict) -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Try Your Own Digit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Upload a single handwritten digit as a PNG, JPG, or JPEG. '
        'The image is converted to an MNIST-style 28x28 grayscale sample before prediction.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is None:
        st.info("Upload an image to see the CNN prediction, confidence, and processed preview.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    original_image = Image.open(uploaded_file)
    processed_input, processed_preview = prepare_uploaded_image(original_image)
    probabilities = model.predict(processed_input, verbose=0)[0]
    predicted_digit = int(probabilities.argmax())
    confidence = float(probabilities[predicted_digit])

    st.markdown(
        f"""
        <div class="prediction-banner">
            <div>Predicted Digit</div>
            <div class="prediction-digit">{predicted_digit}</div>
            <div>Confidence: {confidence:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_col, preview_col = st.columns(2)
    with image_col:
        st.image(original_image, caption="Uploaded image", use_container_width=True)
    with preview_col:
        st.image(processed_preview, caption="Processed 28x28 preview", use_container_width=True)

    probability_df = pd.DataFrame({"Digit": list(range(10)), "Probability": probabilities})
    st.bar_chart(probability_df, x="Digit", y="Probability", use_container_width=True)
    st.caption(f"The model type is {metadata['model_type']}.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_experiment_panel(metadata: dict) -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Experiment Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The build artifacts below summarize model behavior, training quality, '
        'and the kinds of mistakes the CNN still makes.</div>',
        unsafe_allow_html=True,
    )

    tab_overview, tab_errors, tab_metrics, tab_model = st.tabs(["Overview", "Error Analysis", "Metrics", "Model Details"])

    with tab_overview:
        if HISTORY_PLOT_PATH.exists():
            st.image(str(HISTORY_PLOT_PATH), caption="Training accuracy and loss", use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion matrix", use_container_width=True)
        with col_b:
            if FEATURE_MAPS_PATH.exists():
                st.image(str(FEATURE_MAPS_PATH), caption="Conv1 feature maps", use_container_width=True)

    with tab_errors:
        left, right = st.columns(2)
        with left:
            if MISCLASSIFIED_PLOT_PATH.exists():
                st.image(str(MISCLASSIFIED_PLOT_PATH), caption="Misclassified samples", use_container_width=True)
        with right:
            if SAMPLE_PREDICTION_PATH.exists():
                st.image(str(SAMPLE_PREDICTION_PATH), caption="Sample prediction from the test set", use_container_width=True)
        st.markdown(
            f'<div class="small-muted">Misclassified test samples recorded in metadata: {metadata["misclassified_samples"]}</div>',
            unsafe_allow_html=True,
        )

    with tab_metrics:
        if CLASSIFICATION_REPORT_PATH.exists():
            st.dataframe(load_dataframe(CLASSIFICATION_REPORT_PATH), use_container_width=True, hide_index=True)
        if HISTORY_CSV_PATH.exists():
            st.markdown("#### Epoch History")
            st.dataframe(load_dataframe(HISTORY_CSV_PATH), use_container_width=True, hide_index=True)

    with tab_model:
        st.json(metadata)
        if MODEL_SUMMARY_PATH.exists():
            st.code(MODEL_SUMMARY_PATH.read_text(encoding="utf-8"), language="text")

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    render_styles()

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("Model artifacts are missing. Run `python train.py` before launching the app.")
        return

    metadata = load_metadata()
    model = load_model()
    render_hero(metadata)
    render_metric_grid(metadata)

    left_column, right_column = st.columns([0.9, 1.1], gap="large")
    with left_column:
        render_prediction_panel(model, metadata)
    with right_column:
        render_experiment_panel(metadata)


if __name__ == "__main__":
    main()
