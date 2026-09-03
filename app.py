"""
app.py
------
AI Research Paper Category Classifier
Streamlit Web Application — IBM Edunet Foundation Project
"""

import json
import re
from pathlib import Path

import joblib
import streamlit as st
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"
META_PATH = BASE_DIR / "models" / "metadata.json"

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
joblib.load(MODEL_PATH)
# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Paper Category Classifier",
    page_icon="🔬",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper — text cleaning (must match train_model.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Load model + metadata (cached so they load once per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML pipeline…")
def load_model():
    if not MODEL_PATH.exists():
        return None, (
            f"Model file not found at `{MODEL_PATH}`.\n\n"
            "Please run `python train_model.py` first to train and save the model."
        )
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline, None
    except Exception as exc:
        return None, f"Error loading model: {exc}"
        


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — model & dataset info
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(meta: dict) -> None:
    with st.sidebar:
        st.title("ℹ️ Model Information")
        st.divider()

        if meta:
            st.subheader("📊 Dataset")
            st.write(f"**Name:** {meta.get('dataset', '—')}")
            st.write(f"**Target:** {meta.get('target_column', '—')}")
            st.write(f"**Problem:** {meta.get('problem_type', '—')}")
            st.write(f"**Categories:** {meta.get('n_classes', '—')}")
            st.write(f"**Features:** {meta.get('text_features', '—')}")

            st.divider()
            st.subheader("🤖 Best Model")
            st.success(meta.get("best_model", "—"))

            metrics = meta.get("metrics", {})
            if metrics:
                st.subheader("📈 Evaluation Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy",  f"{metrics.get('accuracy',  0):.4f}")
                    st.metric("Recall",    f"{metrics.get('recall',    0):.4f}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                    st.metric("F1-Score",  f"{metrics.get('f1',        0):.4f}")

            st.divider()
            st.subheader("🔢 Sample Counts")
            st.write(f"**Training:** {meta.get('n_train', '—'):,}")
            st.write(f"**Test:**     {meta.get('n_test', '—'):,}")

            all_results = meta.get("all_model_results", [])
            if all_results:
                st.divider()
                st.subheader("📋 All Models Compared")
                for r in all_results:
                    with st.expander(r["name"]):
                        st.write(f"Accuracy : {r['accuracy']:.4f}")
                        st.write(f"Precision: {r['precision']:.4f}")
                        st.write(f"Recall   : {r['recall']:.4f}")
                        st.write(f"F1-Score : {r['f1']:.4f}")

            if "sampling_strategy" in meta:
                st.divider()
                st.caption(f"**Sampling:** {meta['sampling_strategy']}")
        else:
            st.info(
                "No metadata found.\n\n"
                "Run `python train_model.py` to generate the model and metadata."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    meta     = load_metadata()
    pipeline, load_error = load_model()

    render_sidebar(meta)

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔬 AI Research Paper Category Classifier")
    st.markdown(
        "Predict the **category** of a research paper using machine learning "
        "and TF-IDF based text classification.\n\n"
        "_Trained on the arXiv scientific dataset · IBM Edunet Foundation Project_"
    )
    st.divider()

    # ── Model load error ──────────────────────────────────────────────────────
    if load_error:
        st.error(load_error)
        st.stop()

    # ── Input form ───────────────────────────────────────────────────────────
    st.subheader("📝 Enter Paper Details")

    title_input = st.text_input(
        label="Research Paper Title",
        placeholder="e.g. Attention Is All You Need",
        help="Enter the full title of the research paper.",
    )

    summary_input = st.text_area(
        label="Research Paper Abstract / Summary",
        placeholder=(
            "Paste the abstract or summary of the paper here. "
            "More text generally gives a better prediction."
        ),
        height=200,
        help="Paste the abstract or summary of the research paper.",
    )

    predict_btn = st.button("🚀 Predict Category", type="primary", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if predict_btn:
        # Validate inputs
        if not title_input.strip():
            st.warning("⚠️ Please enter the research paper title.")
            st.stop()
        if not summary_input.strip():
            st.warning("⚠️ Please enter the research paper abstract/summary.")
            st.stop()

        # Build text exactly as during training
        combined_text = (
            clean_text(title_input) + " " + clean_text(summary_input)
        )

        with st.spinner("Classifying …"):
            try:
                prediction = pipeline.predict([combined_text])[0]  # type: ignore[union-attr]

                # Probability / confidence (if supported)
                confidence = None
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba([combined_text])[0]  # type: ignore[union-attr]
                    confidence = float(proba.max())

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        # ── Result display ────────────────────────────────────────────────────
        st.divider()
        st.subheader("🎯 Prediction Result")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"**Predicted Category**\n\n# {prediction}")
        with col2:
            if confidence is not None:
                pct = confidence * 100
                color = "green" if pct >= 60 else "orange" if pct >= 35 else "red"
                st.metric(
                    label="Model Confidence",
                    value=f"{pct:.1f}%",
                    help=(
                        "Probability that the model assigns to this prediction. "
                        "Higher is more confident."
                    ),
                )
                st.progress(confidence)
            else:
                st.info("Confidence score not available for this model type.")

        # ── Show input summary ────────────────────────────────────────────────
        with st.expander("📄 Input summary"):
            st.write(f"**Title:** {title_input}")
            st.write(f"**Abstract (first 300 chars):** {summary_input[:300]}…"
                     if len(summary_input) > 300 else
                     f"**Abstract:** {summary_input}")
            if meta.get("best_model"):
                st.write(f"**Model used:** {meta['best_model']}")

    # ── Footer hint ───────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "💡 Tip: The more descriptive the title and abstract, "
        "the more accurate the prediction will be."
    )


if __name__ == "__main__":
    main()

