# 🏥 Clinical Early Warning System  
## Can AI Save Lives? — Deep Learning for ICU Patient Deterioration Prediction

Predicting patient deterioration risk (**sepsis, cardiac arrest, ICU transfer, mortality**) using three generations of deep learning models — from a baseline Deep Neural Network (DNN) to **ClinicalBERT** fine-tuned on clinical narratives.

---

# 📌 Overview

This project builds a progressively more sophisticated AI pipeline that monitors ICU patients and flags those at risk of deterioration **before a clinical crisis occurs**.

The project mirrors the real-world evolution of deep learning in healthcare:

1. **Generation 1 — Feedforward Neural Networks**
2. **Generation 2 — Sequential Deep Learning (LSTM/GRU)**
3. **Generation 3 — Transformer-based Clinical NLP (ClinicalBERT)**

The goal is not only predictive performance, but also understanding:

- Why architectures evolved
- What limitations each model solves
- Which models are practical for real clinical deployment

---

# 📊 Dataset

**Dataset:** Patient Survival Prediction (Kaggle)

- **64,199 ICU admissions**
- **8.6% mortality rate**
- Highly imbalanced clinical dataset

The dataset contains:

- Demographics
- Vital signs
- Laboratory measurements
- ICU stay information
- Survival outcome labels

---

# 📂 Project Structure

```bash
├── Generation1_EWS.ipynb
├── Generation2_EWS.ipynb
├── Generation3_EWS.ipynb
├── EWS_Report_PartB.docx
└── README.md
```

---

# 🧠 The Three Generations

# 🔹 Generation 1 — DNN Baseline

Establishes what a standard feedforward neural network can and cannot do on structured ICU tabular data.

## Features

- Median Imputation
- StandardScaler normalization
- SMOTE for class imbalance
- SGD vs Adam optimizer comparison
- Dropout regularization
- Batch Normalization
- Binary Cross-Entropy Loss
- Recall-focused evaluation

## Key Insight

Simple DNNs can learn meaningful clinical patterns but struggle to model temporal patient deterioration trends.

---

# 🔹 Generation 2 — Sequential Models (Patient Timelines)

Transforms patient records into sequential medical timelines:

```text
Hour 1 vitals → Day 1 vitals
```

This allows the network to model patient progression over time.

## Models

- LSTM
- GRU
- Bidirectional LSTM

## Features

- Time-series sequence construction
- Class weighting for imbalance handling
- Temporal learning comparison
- Training time benchmarking
- Real-time deployment analysis

## Key Insight

GRU achieved the best balance between:

- Recall
- Speed
- Real-time deployment feasibility

---

# 🔹 Generation 3 — ClinicalBERT (Clinical NLP)

Structured clinical measurements were converted into synthetic clinical narratives and processed using a transformer-based language model.

## Model

- `emilyalsentzer/Bio_ClinicalBERT`

## Experiments

### 1. Frozen ClinicalBERT
- Only classification head trained

### 2. Full Fine-Tuning
- Entire transformer updated on ICU task

## Features

- HuggingFace Transformers
- Attention extraction
- Attention heatmap visualization
- Per-class performance analysis
- Clinical text tokenization

## Key Insight

Full fine-tuning dramatically improved Recall but increased training cost.

---

# 📈 Results

| Model | Accuracy | Precision | Recall ⭐ | F1 Score | Train Time |
|---|---|---|---|---|---|
| DNN (Adam) | 0.8618 | 0.3317 | 0.5681 | 0.4189 | — |
| DNN (SGD) | 0.8301 | 0.2921 | 0.6583 | 0.4046 | — |
| LSTM | 0.7721 | 0.2445 | 0.7652 | 0.3706 | 59.8s |
| GRU ⭐ | 0.7735 | 0.2482 | 0.7799 | 0.3765 | 74.7s |
| Bi-LSTM | 0.7820 | 0.2523 | 0.7568 | 0.3784 | 97.2s |
| ClinicalBERT (Frozen) | 0.9017 | 0.3030 | 0.0952 | 0.1449 | 279.2s |
| ClinicalBERT (Full) | 0.7067 | 0.1841 | 0.6857 | 0.2903 | 645.8s |

---

# ⭐ Why Recall Matters Most

In healthcare, missing a critically deteriorating patient (**False Negative**) can be life-threatening.

A false alarm (**False Positive**) only costs a few minutes of clinical review.

Therefore:

```text
Recall > Accuracy
```

A model predicting "survived" for every patient would achieve:

```text
91.4% Accuracy
```

while identifying:

```text
0% deteriorating patients
```

---

# ✅ Recommended Deployment Strategy

## Real-Time ICU Alerting
### → GRU Model

Why?

- High Recall
- Faster inference
- Suitable for live monitoring systems

## Offline Risk Stratification
### → Fully Fine-Tuned ClinicalBERT

Why?

- Better contextual reasoning
- Stronger semantic understanding
- Useful for physician decision support

---

# ⚙️ Setup

All notebooks are designed for:

- Google Colab
- NVIDIA T4 GPU

## Install Dependencies

```python
pip install transformers==4.40.0 bertviz torch imbalanced-learn scikit-learn
```

---

# 🔄 Session Reset Strategy

Each notebook automatically saves:

- Preprocessed datasets
- Trained models
- Tokenized clinical narratives

to Google Drive.

## After Colab Disconnect

Simply run:

1. Install cell
2. Import + Drive mount cell
3. Reload checkpoint cell

Everything reloads in ~5 seconds without retraining.

---

# 💾 Google Drive File Layout

```bash
MyDrive/EWS_Project/
├── data/
│   ├── train.csv
│   ├── X_train_bal.npy
│   ├── X_seq_train.npy
│   └── clinical_notes.pkl
│
└── models/
    ├── dnn_adam.keras
    ├── lstm_model.keras
    ├── gru_model.keras
    ├── bilstm_model.keras
    ├── clinicalbert_frozen.pt
    └── clinicalbert_full.pt
```

---

# 🏗️ Key Design Decisions

## Why Recall Instead of Accuracy?

The dataset is heavily imbalanced.

Accuracy alone becomes clinically misleading.

Recall directly measures:

> “How many deteriorating patients were successfully identified?”

---

## Why `class_weight` Instead of SMOTE for Sequences?

SMOTE works naturally on 2D tabular data.

However, sequential tensors are 3D:

```text
(samples, timesteps, features)
```

Synthetic interpolation in this space is geometrically unreliable.

Using:

```python
class_weight
```

preserves real temporal structure while handling imbalance safely.

---

## Why Pseudo Clinical Notes?

The dataset contains structured measurements rather than free-text notes.

To demonstrate a full clinical NLP pipeline:

- Numerical features were transformed into narrative-style text
- ClinicalBERT processed these narratives

This approach is commonly used in clinical AI research when raw clinical notes are unavailable.

---

## Why Not Deploy Bidirectional LSTM?

Bidirectional LSTM requires future timesteps:

```text
t+1, t+2, ...
```

to make predictions at time `t`.

Real-time ICU monitoring systems do not have access to future patient data.

Therefore:

❌ Not suitable for live deployment

---

# 🛠️ Tech Stack

| Component | Library |
|---|---|
| DNN / LSTM / GRU | TensorFlow / Keras |
| ClinicalBERT | HuggingFace Transformers + PyTorch |
| Preprocessing | scikit-learn, imbalanced-learn |
| Visualization | matplotlib, seaborn |
| Attention Visualization | bertviz + custom heatmaps |
| Environment | Google Colab (T4 GPU) |

---

## Accountability

This system is designed to:

✅ Assist clinicians  
❌ Replace clinicians

Every alert must be reviewed by medical professionals before intervention.

---
