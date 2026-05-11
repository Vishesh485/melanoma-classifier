# Melanoma Skin Cancer Classifier — Streamlit App

A web frontend for the MobileNetV2 melanoma classifier from the
`Skin_Lesion_Classification.ipynb` notebook. Upload a dermoscopic image and
the app predicts **benign** vs **malignant**, plus shows the full evaluation
report (precision, recall, F1, confusion matrix, charts).

## Project structure
```
melanoma_app/
├── app.py                              # Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                           # this file
└── melanoma_cancer_model.keras         # ← place your model file here
```

## Setup

### 1. Get the model file
Your uploaded `melanoma_cancer_model_keras.zip` **is the `.keras` file** —
Keras 3 saves models as zip archives. Just rename it:

```bash
mv melanoma_cancer_model_keras.zip melanoma_cancer_model.keras
```

Place it in the same folder as `app.py`.

> The app also accepts the `.zip` directly or an unzipped `model_extracted/`
> folder — it tries all three locations in order.

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`.

## What's inside

The app has four tabs:

1. **🖼️  Predict** — drag-and-drop image upload, prediction with confidence
   banner, class-probability bar chart, and clinical-style guidance text.
2. **📊 Metrics** — accuracy, macro/weighted precision/recall/F1, per-class
   metric table, and the full scikit-learn classification report.
3. **🧮 Confusion Matrix** — interactive heatmaps (counts + normalised %),
   plus a TP / FP / FN / TN breakdown.
4. **📈 Charts** — grouped bar chart of per-class metrics, class
   distribution pie, correct-vs-incorrect stacked bars, and a performance
   radar chart.

All metric values are the ones reported in your notebook on the held-out
test set (1,000 images, 500 per class):

| Class      | Precision | Recall | F1     | Support |
|------------|-----------|--------|--------|---------|
| benign     | 0.8827    | 0.9180 | 0.9000 | 500     |
| malignant  | 0.9146    | 0.8780 | 0.8959 | 500     |
| **Accuracy** | | | **0.90** | **1000** |

## Preprocessing
The app uses the **same preprocessing as training**: resize to 224×224, RGB
conversion, and `rescale=1/255` — matching the `ImageDataGenerator` in your
notebook.

## Note
This is an educational project, not a medical device. The model can be
wrong — never use predictions for diagnosis.
