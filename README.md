# 🎭 DeepFER: Real-Time Facial Emotion Recognition Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![ResNet50](https://img.shields.io/badge/Model-ResNet50%20Transfer%20Learning-blue?style=for-the-badge)](https://keras.io)
[![Grad-CAM](https://img.shields.io/badge/Interpretability-Grad--CAM%20Heatmaps-purple?style=for-the-badge)](https://keras.io)
[![Live Webcam](https://img.shields.io/badge/Inference-Real--Time%20Webcam-green?style=for-the-badge)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end Computer Vision system trained on ~35,000 images from the FER-2013 benchmark to classify human facial expressions into 7 discrete emotional states. Combines class-weighted deep convolutional neural networks, transfer learning with ResNet50, visual explainability with Grad-CAM, and a real-time OpenCV webcam application (`app.py`).

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [The Emotion Taxonomy](#-the-emotion-taxonomy)
- [Data Engineering & Class Imbalance](#-data-engineering--class-imbalance)
- [Model Architecture & Benchmarks](#-model-architecture--benchmarks)
- [Explainability with Grad-CAM](#-explainability-with-grad-cam)
- [Real-Time Inference App](#-real-time-inference-app)
- [Repository Structure](#-repository-structure)
- [Installation & Execution](#-installation--execution)
- [Author & Connect](#-author--connect)

---

## 🚀 Executive Overview
Automated emotion recognition has transformative applications across mental health monitoring, human-computer interaction, driver alertness systems, and customer sentiment analytics.

**DeepFER** addresses core challenges in real-world emotion classification:
- **Severe Class Imbalance:** Neutralizes the 16:1 ratio between Happy and Disgust categories using algorithmic class weights and data augmentation.
- **Pixel Ambiguity:** Resolves subtle boundary overlap between sadness, fatigue, and neutral expressions.
- **Model Explainability:** Integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to verify attention on expressive facial muscle groups.
- **Real-Time Deployment:** Couples Haar Cascade frontal face localization with deep inference at 30+ FPS.

---

## 🏗️ System Architecture

```text
[ Live Webcam Feed / Input Image ]
                │
                ▼
┌────────────────────────────────────────────────────────┐
│             Face Detection & Localization              │
│  Haar Cascade Frontal Face Classifier (XML)            │
│  Extracts Bounding Box & Crops Facial Region of Int.   │
└────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────┐
│            Preprocessing & Normalization               │
│  • Grayscale conversion & 48x48 / 224x224 resize       │
│  • Normalization [0, 1] & Tensor shape formatting      │
└────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────────┐
│          Deep Residual Network (ResNet50 / CNN)        │
│  • Deep Convolutional Feature Extraction               │
│  • Global Average Pooling -> Dense -> Dropout          │
│  • Softmax Probabilities across 7 Emotion Classes      │
└────────────────────────────────────────────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  Emotion Box │  │   Grad-CAM   │
│  HUD Overlay │  │ Visual Heat  │
└──────────────┘  └──────────────┘
```

---

## 🏷️ The Emotion Taxonomy
Classifies facial expressions into 7 affective categories:
1. **Angry 😠**
2. **Disgust 🤢**
3. **Fear 😨**
4. **Happy 😊**
5. **Sad 😢**
6. **Surprise 😲**
7. **Neutral 😐**

---

## 🧠 Model Architecture & Benchmarks

| Architecture | Validation Accuracy | Disgust F1-Score | Inference Latency |
|---|---|---|---|
| **ResNet50 Transfer Learning (Best)** | **67.5%** | **59.8%** | **~24ms / frame** |
| Custom 4-Layer CNN Baseline | 54.2% | 31.2% | ~12ms / frame |
| MobileNetV2 | 64.1% | 51.0% | ~18ms / frame |

---

## 🔍 Explainability with Grad-CAM
To guarantee that predictions are not driven by background noise, hair styling, or illumination artifacts, **Grad-CAM heatmaps** confirm:
- **Happy Predictions:** Model consistently attends to the zygomaticus major (smile lines, mouth curvature).
- **Surprise Predictions:** Model focuses on eyebrow elevation and open oral cavities.
- **Anger Predictions:** Model concentrates on furrowed brows and compressed lips.

---

## 📁 Repository Structure
```text
├── DeepFER.ipynb                          # Training, data augmentation & Grad-CAM evaluation notebook
├── app.py                                 # Real-time webcam inference application
├── haarcascade_frontalface_default.xml   # Pre-trained OpenCV Haar Cascade detector
├── requirements.txt                       # Dependencies (opencv-python, tensorflow, numpy)
├── myphoto.jpg                            # Sample test image
└── README.md                              # Comprehensive project documentation
```

---

## 🛠️ Installation & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/abhik99/DeepFER-Facial-Emotion-Recognition-Using-Deep-Learning.git
cd DeepFER-Facial-Emotion-Recognition-Using-Deep-Learning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Real-Time Webcam Classifier
```bash
python app.py
```
*Press `q` on the video window to quit.*

---

## 👨‍💻 Author & Connect

**Abhishek Kumar (Abhi)**  
*Data Scientist & Analytics Engineer*

- 🌐 **Portfolio:** [abhicodes.online](https://abhicodes.online)
- 💼 **LinkedIn:** [linkedin.com/in/abhik99](https://linkedin.com/in/abhik99)
- 🐙 **GitHub:** [@abhik99](https://github.com/abhik99)
- 📧 **Email:** [akdevil27boy@gmail.com](mailto:akdevil27boy@gmail.com)

---
*Distributed under the MIT License.*
