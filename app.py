"""
Melanoma Skin Cancer Classification — Streamlit Frontend
========================================================
Upload a skin lesion image and the app predicts benign vs malignant
using a MobileNetV2-based model. Also shows full evaluation metrics
(precision, recall, F1, confusion matrix, charts) from the notebook.
"""

import os
import io
import json
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Melanoma Skin Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom styling
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecf3 100%);
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .big-prediction {
        padding: 28px;
        border-radius: 14px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: white;
        margin: 16px 0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .benign-bg { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .malignant-bg { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .info-card {
        background: #f8fafc;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin: 8px 0;
    }
    h1, h2, h3 { color: #1e293b; }
    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 14px;
        border-radius: 8px;
        margin: 12px 0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# Constants — class names + cached metrics from your notebook
# ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["benign", "malignant"]
IMG_SIZE = (224, 224)

# Test-set metrics (1000 images: 500 benign, 500 malignant) — from notebook cell 14
NOTEBOOK_METRICS = {
    "per_class": {
        "benign":    {"precision": 0.8827, "recall": 0.9180, "f1": 0.9000, "support": 500},
        "malignant": {"precision": 0.9146, "recall": 0.8780, "f1": 0.8959, "support": 500},
    },
    "overall": {
        "accuracy": 0.90,
        "macro_precision": 0.8986,
        "macro_recall": 0.898,
        "macro_f1": 0.8980,
        "weighted_precision": 0.8986,
        "weighted_recall": 0.898,
        "weighted_f1": 0.8980,
    },
    # Derived from precision/recall on 500/class:
    # benign recall 0.918  → 459 TP / 41 FN ; benign precision 0.8827 → 459/(459+61) ✓
    # malignant recall 0.878 → 439 TP / 61 FN ; malignant precision 0.9146 → 439/(439+41) ✓
    "confusion_matrix": np.array([[459, 41], [61, 439]]),
}

# ──────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the Keras model from any of the supported locations."""
    candidates = [
        "melanoma_cancer_model.keras",
        "model/melanoma_cancer_model.keras",
        "melanoma_cancer_model_keras.zip",   # the zip is actually a valid .keras file
    ]
    for path in candidates:
        if os.path.exists(path):
            return tf.keras.models.load_model(path), path

    # Fallback: rebuild from extracted folder (config.json + model.weights.h5)
    extracted_dir = "model_extracted"
    if os.path.isdir(extracted_dir):
        with open(os.path.join(extracted_dir, "config.json")) as f:
            cfg = json.load(f)
        model = tf.keras.models.model_from_json(json.dumps(cfg))
        model.load_weights(os.path.join(extracted_dir, "model.weights.h5"))
        return model, extracted_dir

    raise FileNotFoundError(
        "Couldn't find the model. Place `melanoma_cancer_model.keras` "
        "(or the original .zip) next to app.py."
    )

# ──────────────────────────────────────────────────────────────────────
# Image preprocessing — matches training pipeline (rescale 1/255)
# ──────────────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(pil_img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, pil_img: Image.Image):
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES[idx],
        "confidence": float(np.max(probs)),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }

# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Melanoma Classifier")
    st.markdown("---")
    st.markdown("### About")
    st.write(
        "A deep-learning classifier for melanoma skin cancer detection. "
        "Built on **MobileNetV2** pretrained on ImageNet, fine-tuned on "
        "10,000 dermoscopic images."
    )
    st.markdown("### Model")
    st.code(
        "Base   : MobileNetV2 (frozen)\n"
        "Input  : 224 × 224 × 3\n"
        "Head   : BN → Dense 512 ReLU\n"
        "         → Dropout 0.5\n"
        "         → Dense 2 Softmax\n"
        "Loss   : categorical_crossentropy\n"
        "Optim. : Adam (lr=1e-4)",
        language="text",
    )
    st.markdown("### Test-set performance")
    st.metric("Accuracy", "90.0%")
    st.metric("Macro F1", "0.898")
    st.markdown("---")
    st.caption("Dataset: hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")

# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────
st.title("🔬 Melanoma Skin Cancer Classifier")
st.markdown(
    "Upload a dermoscopic image to classify it as **benign** or **malignant**, "
    "and review the model's full evaluation report."
)

st.markdown(
    '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This is an '
    "educational demo, not a medical device. Predictions must not be used "
    "for diagnosis. Always consult a qualified dermatologist for any skin "
    "lesion of concern.</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_cm, tab_charts = st.tabs(
    ["🖼️  Predict", "📊 Metrics", "🧮 Confusion Matrix", "📈 Charts"]
)

# ──────────────────────────────────────────────────────────────────────
# TAB 1 — Prediction
# ──────────────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Upload a skin lesion image")
    uploaded = st.file_uploader(
        "Drag and drop or browse — JPG / PNG / JPEG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        try:
            model, _ = load_model()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        col1, col2 = st.columns([1, 1.2])

        with col1:
            img = Image.open(uploaded)
            st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)
            st.caption(f"Original size: {img.size[0]} × {img.size[1]} px  →  resized to 224 × 224 for inference")

        with col2:
            with st.spinner("Running inference…"):
                result = predict(model, img)

            # Big prediction banner
            css_class = "malignant-bg" if result["class"] == "malignant" else "benign-bg"
            label = result["class"].upper()
            emoji = "⚠️" if result["class"] == "malignant" else "✅"
            st.markdown(
                f'<div class="big-prediction {css_class}">{emoji} Prediction: {label}<br>'
                f'<span style="font-size:18px;font-weight:500;">Confidence: '
                f'{result["confidence"]*100:.2f}%</span></div>',
                unsafe_allow_html=True,
            )

            # Probability bars
            st.markdown("##### Class probabilities")
            prob_df = pd.DataFrame({
                "Class": list(result["probs"].keys()),
                "Probability": [v * 100 for v in result["probs"].values()],
            })
            fig = px.bar(
                prob_df, x="Probability", y="Class", orientation="h",
                text=prob_df["Probability"].round(2).astype(str) + "%",
                color="Class",
                color_discrete_map={"benign": "#10b981", "malignant": "#ef4444"},
                range_x=[0, 100],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                showlegend=False,
                height=240,
                margin=dict(l=0, r=20, t=10, b=0),
                xaxis_title="Probability (%)",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            if result["class"] == "malignant":
                st.warning(
                    "🩺 The model classifies this lesion as **malignant**. "
                    "Strongly recommend evaluation by a dermatologist."
                )
            else:
                st.success(
                    "🌿 The model classifies this lesion as **benign**. "
                    "Routine monitoring is still advised; consult a doctor "
                    "if the lesion changes."
                )

            with st.expander("Raw model output"):
                st.json(result["probs"])
    else:
        st.info("👆 Upload an image above to get a prediction.")

# ──────────────────────────────────────────────────────────────────────
# TAB 2 — Metrics
# ──────────────────────────────────────────────────────────────────────
with tab_metrics:
    st.subheader("Test-set performance")
    st.caption("Evaluated on 1,000 held-out images (500 benign + 500 malignant)")

    # Overall metrics row
    ov = NOTEBOOK_METRICS["overall"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",        f"{ov['accuracy']*100:.1f}%")
    c2.metric("Macro Precision", f"{ov['macro_precision']:.4f}")
    c3.metric("Macro Recall",    f"{ov['macro_recall']:.4f}")
    c4.metric("Macro F1-score",  f"{ov['macro_f1']:.4f}")

    st.markdown("### Per-class metrics")
    per_class = NOTEBOOK_METRICS["per_class"]
    metric_df = pd.DataFrame({
        "Class":     list(per_class.keys()),
        "Precision": [per_class[c]["precision"] for c in per_class],
        "Recall":    [per_class[c]["recall"]    for c in per_class],
        "F1-score":  [per_class[c]["f1"]        for c in per_class],
        "Support":   [per_class[c]["support"]   for c in per_class],
    })
    st.dataframe(
        metric_df.style.format({
            "Precision": "{:.4f}",
            "Recall":    "{:.4f}",
            "F1-score":  "{:.4f}",
        }).background_gradient(
            subset=["Precision", "Recall", "F1-score"], cmap="Greens"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Classification report")
    report_text = (
        "              precision    recall  f1-score   support\n\n"
        "      benign       0.88      0.92      0.90       500\n"
        "   malignant       0.91      0.88      0.90       500\n\n"
        "    accuracy                           0.90      1000\n"
        "   macro avg       0.90      0.90      0.90      1000\n"
        "weighted avg       0.90      0.90      0.90      1000\n"
    )
    st.code(report_text, language="text")

    st.markdown("### Weighted vs Macro averages")
    avg_df = pd.DataFrame({
        "Average": ["Macro", "Weighted"],
        "Precision": [ov["macro_precision"], ov["weighted_precision"]],
        "Recall":    [ov["macro_recall"],    ov["weighted_recall"]],
        "F1-score":  [ov["macro_f1"],        ov["weighted_f1"]],
    })
    st.dataframe(
        avg_df.style.format({
            "Precision": "{:.4f}",
            "Recall":    "{:.4f}",
            "F1-score":  "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ──────────────────────────────────────────────────────────────────────
# TAB 3 — Confusion Matrix
# ──────────────────────────────────────────────────────────────────────
with tab_cm:
    st.subheader("Confusion matrix")
    st.caption("Rows = true label · Columns = predicted label")

    cm = NOTEBOOK_METRICS["confusion_matrix"]
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Counts**")
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=CLASS_NAMES, y=CLASS_NAMES,
            text=cm, texttemplate="%{text}",
            colorscale="Blues",
            textfont={"size": 22, "color": "white"},
            showscale=True,
        ))
        fig.update_layout(
            xaxis_title="Predicted", yaxis_title="True",
            height=420, margin=dict(l=40, r=20, t=20, b=40),
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Normalised (%)**")
        fig = go.Figure(data=go.Heatmap(
            z=cm_pct,
            x=CLASS_NAMES, y=CLASS_NAMES,
            text=np.round(cm_pct, 2),
            texttemplate="%{text}%",
            colorscale="Purples",
            textfont={"size": 22, "color": "white"},
            showscale=True,
        ))
        fig.update_layout(
            xaxis_title="Predicted", yaxis_title="True",
            height=420, margin=dict(l=40, r=20, t=20, b=40),
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Breakdown")
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("True Negatives (benign → benign)",       int(tn))
    b2.metric("False Positives (benign → malignant)",   int(fp))
    b3.metric("False Negatives (malignant → benign)",   int(fn))
    b4.metric("True Positives (malignant → malignant)", int(tp))

    st.info(
        f"📌 The model misses **{int(fn)}** malignant lesions out of 500 (false negatives) — "
        f"the most critical error type in clinical screening. It also incorrectly flags "
        f"**{int(fp)}** benign lesions as malignant (false positives)."
    )

# ──────────────────────────────────────────────────────────────────────
# TAB 4 — Charts
# ──────────────────────────────────────────────────────────────────────
with tab_charts:
    st.subheader("Visual breakdown of performance")

    # Per-class metric bar chart
    st.markdown("### Per-class precision / recall / F1")
    per_class = NOTEBOOK_METRICS["per_class"]
    chart_df = pd.DataFrame([
        {"Class": c, "Metric": m.capitalize(), "Value": v}
        for c, vals in per_class.items()
        for m, v in vals.items() if m != "support"
    ])
    fig = px.bar(
        chart_df, x="Class", y="Value", color="Metric",
        barmode="group", text=chart_df["Value"].round(4),
        color_discrete_map={
            "Precision": "#6366f1",
            "Recall":    "#f59e0b",
            "F1":        "#10b981",
        },
        range_y=[0, 1.05],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Class distribution
    st.markdown("### Test-set class distribution")
    col1, col2 = st.columns(2)
    with col1:
        dist_df = pd.DataFrame({
            "Class": list(per_class.keys()),
            "Count": [per_class[c]["support"] for c in per_class],
        })
        fig = px.pie(
            dist_df, names="Class", values="Count", hole=0.55,
            color="Class",
            color_discrete_map={"benign": "#10b981", "malignant": "#ef4444"},
        )
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Correct vs incorrect predictions per class
        cm = NOTEBOOK_METRICS["confusion_matrix"]
        comp_df = pd.DataFrame({
            "Class":     CLASS_NAMES * 2,
            "Outcome":   ["Correct"] * 2 + ["Incorrect"] * 2,
            "Count":     [int(cm[0,0]), int(cm[1,1]), int(cm[0,1]), int(cm[1,0])],
        })
        fig = px.bar(
            comp_df, x="Class", y="Count", color="Outcome",
            barmode="stack", text="Count",
            color_discrete_map={"Correct": "#10b981", "Incorrect": "#ef4444"},
        )
        fig.update_traces(textposition="inside", textfont_size=16, textfont_color="white")
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("### Performance radar")
    categories = ["Precision", "Recall", "F1-score"]
    fig = go.Figure()
    for cls in CLASS_NAMES:
        vals = [per_class[cls]["precision"], per_class[cls]["recall"], per_class[cls]["f1"]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=cls,
            line=dict(color="#10b981" if cls == "benign" else "#ef4444"),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.85, 1.0])),
        height=440, margin=dict(l=40, r=40, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)
