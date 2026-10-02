# 🫁 AI-Powered Chest X-Ray Pneumonia Detection

An end-to-end deep learning system for **pneumonia classification and visual explainability from chest X-ray images**, built with **EfficientNetB3, TensorFlow, Grad-CAM, and Gradio**.

The project improves upon a MobileNetV2 baseline through transfer learning, progressive fine-tuning, class-weighted training, threshold optimization, and visual model interpretation.

> **Note:** This project is intended for academic research, education, and demonstration purposes. It is not a medical diagnostic device and should not be used as a substitute for professional medical evaluation.

---

## 🚀 Project Overview

Chest X-ray classification models can achieve strong accuracy while remaining difficult to interpret.

This project focuses on both:

- **Pneumonia classification**
- **Visual explanation of model predictions**

The system uses **EfficientNetB3** as the feature extraction backbone and combines it with a custom classification head. Grad-CAM is then used to visualize regions that contributed to the model's prediction.

The final application provides an interactive **Gradio interface** where users can upload a chest X-ray and receive a prediction along with visual explanation and model-generated information.

---

## ✨ Key Features

- 🧠 EfficientNetB3-based image classification
- 🔄 Two-phase transfer learning
- 🎯 Fine-tuning of the top EfficientNetB3 layers
- ⚖️ Class-weighted training for imbalanced data
- 📊 Threshold optimization using Youden's J statistic
- 🔥 Grad-CAM visual explainability
- 📍 Activation-based defect localization
- 🫁 Left/right lung region analysis
- 🖥️ Interactive Gradio web interface
- 📋 Structured prediction report
- 📈 Training and validation performance visualization
- 📸 Comparison with the original MobileNetV2 baseline

---

# 📊 Model Performance

The improved EfficientNetB3 model was evaluated against the original MobileNetV2 baseline.

| Metric | MobileNetV2 Baseline | EfficientNetB3 | Improvement |
|---|---:|---:|---:|
| Test Accuracy | 85.74% | **92.95%** | +7.21% |
| AUC-ROC | 0.8900 | **0.9814** | +0.0914 |
| Normal Recall | 69.00% | **91.45%** | +22.45% |
| Pneumonia Recall | 96.00% | **94.62%** | Balanced performance |
| Macro F1-Score | 0.8400 | **0.9300** | +0.0900 |
| False Positives | 72 | **20** | 72.2% reduction |

These values are based on the evaluation results included with this project.

---

# 🏗️ System Architecture

```text
                    Chest X-Ray Image
                           │
                           ▼
              ┌────────────────────────┐
              │ Image Preprocessing     │
              │ 224 × 224 × 3          │
              │ Data Augmentation       │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ EfficientNetB3         │
              │ Pretrained Backbone     │
              └────────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Phase 1 Training          Phase 2 Fine-Tuning
       Frozen Backbone            Top 100 Layers
              │                         │
              └────────────┬────────────┘
                           ▼
              ┌────────────────────────┐
              │ Classification Head    │
              │                        │
              │ GAP                    │
              │ Dense(512) + BN        │
              │ Dropout(0.4)           │
              │ Dense(256) + BN        │
              │ Dropout(0.3)           │
              │ Dense(128) + BN        │
              │ Dropout(0.2)           │
              │ Dense(1, Sigmoid)      │
              └────────────┬───────────┘
                           │
                           ▼
                 Pneumonia Probability
                           │
                  Threshold = 0.60
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Classification             Grad-CAM
                                  Explainability
              │                         │
              └────────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │ Gradio Web Interface│
                │                     │
                │ Prediction          │
                │ Heatmap             │
                │ Localization        │
                │ Report              │
                └─────────────────────┘
