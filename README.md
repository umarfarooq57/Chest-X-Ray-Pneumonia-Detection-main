# 🫁 AI-Powered Chest X-Ray Pneumonia Detection & Defect Localization System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![EfficientNet-B3](https://img.shields.io/badge/Backbone-EfficientNet--B3-brightgreen.svg)]()
[![AUC-ROC 0.9814](https://img.shields.io/badge/AUC--ROC-0.9814-success.svg)]()
[![Test Accuracy 92.95%](https://img.shields.io/badge/Test%20Accuracy-92.95%25-green.svg)]()
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-red.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Production-Grade, Clinically Explainable Deep Learning System for Pediatric Chest Radiograph Analysis.**  
> Built with **EfficientNetB3**, Two-Phase Transfer Learning, **Grad-CAM Visual Defect Localization**, and an automated **Clinical Diagnostic Report Generator**.

---

## 🌟 Executive Summary & Key Highlights

This project addresses one of the most critical challenges in medical deep learning: **moving beyond black-box classification into clinically actionable, interpretable AI**. 

While standard baseline models report superficial accuracies, they suffer from catastrophic false-positive rates on imbalanced pediatric datasets. This project completely re-engineers the pipeline with state-of-the-art vision backbones, progressive fine-tuning, threshold optimization, and visual lesion localization.

### 🏆 Key Performance Metrics (At a Glance)

| Metric | Baseline (Original MobileNetV2) | **Our Improved Model (EfficientNetB3)** | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 85.74% | **92.95%** | **+7.21%** 🚀 |
| **Area Under ROC (AUC-ROC)** | 0.8900 | **0.9814** | **+0.0914** 🎯 |
| **NORMAL Class Recall** | 69.00% | **91.45%** | **+22.45%** 🛡️ |
| **PNEUMONIA Recall** | 96.00% | **94.62%** | Balanced Clinical Trade-off |
| **Macro F1-Score** | 0.8400 | **0.9300** | **+0.0900** 📈 |
| **False Positives (Healthy misdiagnosed)** | **72** | **20** | **72.2% Reduction** ✅ |
| **Visual Explainability** | ❌ None | **✅ Grad-CAM Heatmap + Defect Bounding Boxes** | Clinical Level |

---

## 📸 Visual Showcase & Results

### 1. Head-to-Head Comparison: Original vs. Improved Model
Our model eliminates **52 false-positive diagnoses**, preventing healthy children from undergoing unnecessary and harmful antibiotic treatments.

<p align="center">
  <img src="Results/Screenshot 2026-10-03 025202.png" width="750" alt="Model Comparison Table" />
</p>

---

### 2. Grad-CAM Model Explainability — "Where is the AI Looking?"
Using Gradient-weighted Class Activation Mapping (Grad-CAM) at the `top_activation` layer, the model transparently reveals its decision boundaries.
- **Normal X-Rays:** Model focuses neutrally on clear lung fields without focal activation.
- **Pneumonia X-Rays:** Intense red heat signatures lock precisely onto focal lobar consolidations and inflammatory infiltrates.

<p align="center">
  <img src="Results/Screenshot 2026-10-03 025304.png" width="850" alt="Grad-CAM Visualization" />
</p>

---

### 3. Interactive Clinical Web Application (Gradio)
The system features an end-to-end interactive diagnostic suite:
1. **Defect Localization Screen:** Overlays high-contrast heatmaps and draws automated **red bounding boxes** around detected infection zones with pixel area quantification and anatomical lateralization (*Right Lung vs. Left Lung*).
2. **Clinical Patient Report:** Translates raw model logits into a human-readable pathology summary, severity grading, expected patient symptoms, and actionable clinical recommendations.

<p align="center">
  <img src="Results/Screenshot 2026-10-03 025000.png" width="900" alt="Gradio Defect Localization" />
</p>

<p align="center">
  <img src="Results/Screenshot 2026-10-03 025104.png" width="750" alt="Gradio Clinical Diagnostic Report" />
</p>

---

### 4. Training & Fine-Tuning Convergence
The training protocol was executed across two distinct phases:
- **Phase 1 (Epochs 1–15):** Frozen EfficientNetB3 backbone, training only the custom dense classification head ($LR = 10^{-4}$). Val accuracy surged to **93.58%**.
- **Phase 2 (Epochs 16–35):** Unfreezing the top 100 layers of EfficientNetB3 with a micro-learning rate ($LR = 10^{-5}$) and plateau decay. Validation accuracy peaked at **95.97%** with validation loss decreasing to **0.1291**.

<p align="center">
  <img src="Results/Screenshot 2026-10-03 025702.png" width="900" alt="Training vs Validation Accuracy and Loss" />
</p>

---

### 5. Dataset Analysis & Class Imbalance Handling
The training split exhibits a severe **2.89 : 1** class imbalance (3,875 Pneumonia vs. 1,341 Normal). Left untreated, standard models develop an overwhelming bias toward predicting Pneumonia.

<p align="center">
  <img src="Results/Screenshot 2026-10-03 025604.png" width="850" alt="Dataset Distribution" />
</p>

**Solution:** We computed inverse class frequency weights:
$$\text{Weight}_{\text{Normal}} = \frac{N_{\text{total}}}{2 \times N_{\text{normal}}} \approx 1.94, \quad \text{Weight}_{\text{Pneumonia}} = \frac{N_{\text{total}}}{2 \times N_{\text{pneumonia}}} \approx 0.67$$
This penalized false positives heavily and skyrocketed Normal recall from **69% to 91.45%**.

---

## 🔬 Core Engineering Innovations

```
Raw X-Ray (Input)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Image Preprocessing (224 × 224 × 3)                   │
│  Data Augmentation (Rotation, Zoom, Shifts, Flips)     │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Backbone: EfficientNetB3 (Compound Scaling)           │
│  Phase 1: Feature Extraction (Frozen Base)             │
│  Phase 2: Fine-Tuning (Top 100 Layers Unfrozen)        │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Custom High-Capacity Classification Head              │
│  GAP ➔ Dense(512)+BN+Dropout(0.4)                      │
│      ➔ Dense(256)+BN+Dropout(0.3)                      │
│      ➔ Dense(128)+BN+Dropout(0.2)                      │
│      ➔ Dense(1, Sigmoid)                               │
└────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Optimal Threshold Tuning      │ │ Grad-CAM Engine               │
│ Youden's J Statistic = 0.60   │ │ Target: 'top_activation'      │
│ Balanced Sensitivity/Recall   │ │ Contour Defect Bounding Boxes │
└───────────────────────────────┘ └───────────────────────────────┘
       │                                 │
       └────────────────┬────────────────┘
                        ▼
       ┌─────────────────────────────────┐
       │ Gradio Live Deployment Suite    │
       │ Visual Heatmap + Lesion Boxes   │
       │ Structured Clinical Report      │
       └─────────────────────────────────┘
```

1. **Resolution Upgrade (150×150 ➔ 224×224):** Higher pixel density captures subtle alveolar infiltrates and fine reticular markings that are lost at lower resolutions.
2. **Backbone Migration (MobileNetV2 ➔ EfficientNetB3):** Compound coefficient scaling uniformly balances network depth, width, and resolution, yielding superior feature representations for radiologic textures.
3. **Threshold Optimization via Youden's $J$ Statistic:** Default decision boundary ($0.50$) is suboptimal for imbalanced medical diagnosis. Optimization shifted threshold to **0.60**, striking the optimal balance between clinical sensitivity and specificity.
4. **Anatomical Defect Localization:** Computer vision contours isolate high-activation coordinates, determining whether lesions reside in the **Left Lung Field** or **Right Lung Field**.

---

## 📂 Repository Structure

```plaintext
Chest-X-Ray-Pneumonia-Detection-main/
│
├── Results/                                # Visual artifacts & evaluation logs
│   ├── Screenshot 2026-10-03 025000.png    # Gradio Defect Localization Screen
│   ├── Screenshot 2026-10-03 025104.png    # Gradio Clinical Diagnostic Report
│   ├── Screenshot 2026-10-03 025202.png    # Model Comparison Matrix
│   ├── Screenshot 2026-10-03 025304.png    # 6-Panel Grad-CAM Visualization
│   ├── Screenshot 2026-10-03 025403.png    # Phase 1 Training Logs
│   ├── Screenshot 2026-10-03 025423.png    # Phase 2 Fine-Tuning Logs
│   ├── Screenshot 2026-10-03 025604.png    # Dataset Class Distribution
│   ├── Screenshot 2026-10-03 025621.png    # Sample X-Ray Dataset Grid
│   ├── Screenshot 2026-10-03 025702.png    # Training vs Validation Curves
│   └── Screenshot 2026-09-24 124035.png    # Normal Lung Detection Sample
│
├── chest_xray_improved.ipynb               # Complete end-to-end production notebook
├── model_config.json                       # Saved optimal thresholds & metadata
└── README.md                               # Comprehensive documentation
```

---

## 🚀 Quick Start Guide

### Option 1: Run in Google Colab (Recommended)
1. Upload `chest_xray_improved.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime ➔ Change runtime type ➔ T4 GPU**.
3. Run all cells sequentially. The final cell will generate a public **Gradio Live URL** accessible from any browser or smartphone.

### Option 2: Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/umarfarooq57/Chest-X-Ray-Pneumonia-Detection-main.git
cd Chest-X-Ray-Pneumonia-Detection-main

# 2. Install dependencies
pip install tensorflow gradio opencv-python matplotlib seaborn scikit-learn pillow kagglehub

# 3. Launch Jupyter Notebook
jupyter notebook chest_xray_improved.ipynb
```

---

## 🩺 Clinical & Ethical Notice

> **⚠️ Disclaimer:** This software is designed for academic research, educational exploration, and clinical decision support demonstration. It has not been cleared as a medical device (FDA SaMD / CE-IVD) and should **never** replace certified radiologist diagnosis or clinical consultation.

---

## 👤 Author & Acknowledgments

- **Developer:** [Umar Farooq](https://github.com/umarfarooq57)
- **Dataset:** [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Guangzhou Women and Children’s Medical Center)
- **Architecture:** EfficientNet by Tan & Le (Google Research)

⭐ *If you find this project helpful or educational, please consider giving it a star on GitHub!*
#   C h e s t - X - R a y - P n e u m o n i a - D e t e c t i o n - m a i n  
 