---
title: Potato Leaf Disease Pathology AI
emoji: 🥔
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: mit
---

# 🥔 Precision Foliar Pathology & Deep Learning Architecture Suite
### Multi-Architecture Neural Ensemble · Explainable AI (Grad-CAM) · Dynamic Research Benchmarking

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow_Lite-Quantized_Zoo-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Explainable_AI-Grad--CAM-8B5CF6?style=for-the-badge"/>
  <a href="https://huggingface.co/spaces/raju-ai/potato-leaf-disease-classifier">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-FFD21E?style=for-the-badge"/>
  </a>
  <a href="https://www.kaggle.com/code/rajucode/potato-leaf-disease-classification-efficientnetb3">
    <img src="https://img.shields.io/badge/Kaggle_Notebook-Open_in_Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  </a>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  🚀 <b>Live Hugging Face Production Demo:</b> <a href="https://huggingface.co/spaces/raju-ai/potato-leaf-disease-classifier"><b>huggingface.co/spaces/raju-ai/potato-leaf-disease-classifier</b></a><br/>
  📓 <b>Official Kaggle Research Benchmark Notebook:</b> <a href="https://www.kaggle.com/code/rajucode/potato-leaf-disease-classification-efficientnetb3"><b>rajucode/potato-leaf-disease-classification-efficientnetb3</b></a>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Research Benchmark & Multi-Architecture Zoo](#-research-benchmark--multi-architecture-zoo)
- [Explainable AI (Grad-CAM Saliency Maps)](#-explainable-ai-grad-cam-saliency-maps)
- [Disease Etiology & Classification](#-disease-etiology--classification)
- [Core Platform Features](#-core-platform-features)
- [Project Architecture](#-project-architecture)
- [Dataset Specifications](#-dataset-specifications)
- [Installation & Local Setup](#-installation--local-setup)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)
- [Author & Citations](#-author--citations)
- [License](#-license)

---

## 🌿 Overview

This repository houses an end-to-end **computer vision and foliar pathology suite** designed for clinical diagnosis of potato crop diseases (*Alternaria solani*, *Phytophthora infestans*, and Healthy foliage). 

Moving beyond single-model prototypes, this system introduces a **Multi-Architecture Deep Learning Model Zoo** coupled with a **Tri-Model Soft-Voting Ensemble** and **Gradient-Weighted Class Activation Mapping (Grad-CAM)** to ensure clinical transparency, high generalization, and zero blind spots.

```
       ┌──────────────────────────────────────────────────────────┐
       │                Input Leaf Image (256×256)                │
       └─────────────────────────────┬────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│   DenseNet-121   │       │  ConvNeXt-Tiny   │       │ EfficientNetV2-S │
│ (Weight: 0.45)   │       │  (Weight: 0.35)  │       │ (Weight: 0.20)   │
└────────┬─────────┘       └─────────┬────────┘       └─────────┬────────┘
         │                           │                          │
         └───────────────────────────┼──────────────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Soft-Voting Ensemble Engine    │
                   │   + Temperature Scaling (T=0.7)   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │  Clinical Pathology Report +      │
                   │  Grad-CAM Visual Saliency Map     │
                   └───────────────────────────────────┘
```

---

## 🔬 Research Benchmark & Multi-Architecture Zoo

All models were trained under rigorous conditions using stratified cross-validation on the held-out test split (405 images). The platform allows users to switch between architectures dynamically or use the recommended soft-voting ensemble:

| Architecture | Paradigm | Parameters | Test Accuracy | Macro-F1 | Empirical Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tri-Model Ensemble** | **Weighted Soft-Voting** | **56.0M** | **100.0%** (Sample) | **100.0%** | **Consensus engine eliminating individual model blind spots** |
| **DenseNet-121** | Dense Feature Reuse | 7.03M | **99.75%** | **99.75%** | Preserves high-frequency lesion edge boundaries |
| **ConvNeXt-Tiny** | Modernized Pure ConvNet | 27.8M | **99.01%** | **99.01%** | 7×7 depthwise convolutions capturing broad leaf context |
| **EfficientNetV2-S** | Neural Architecture Search | 20.3M | **98.77%** | **98.76%** | Fused-MBConv layers for high compute efficiency |
| **ResNet-50** | Deep Residual Learning | 23.6M | 96.30% | 96.24% | Robust baseline architecture |
| **MobileNetV3-Large** | Ultra-Lightweight Edge | 4.20M | 94.81% | 94.75% | Ultra-fast low-power edge deployment |

---

## 🔍 Explainable AI (Grad-CAM Saliency Maps)

To ensure clinical trustworthiness for agronomists and plant pathologists, each prediction is accompanied by a **Gradient-Weighted Class Activation Map (Grad-CAM)**:

- **Target Conv Layer:** `conv5_block16_2_conv` / pen-ultimate convolutional feature extractor.
- **Opacity Blend Controls:** Interactive slider allowing seamless transitions from Raw Leaf (0%), Overlay Blend (50%), to Full Activation Heatmap (100%).
- **Pathological Feature Attribution:** Localizes concentric target spots in Early Blight and water-soaked spreading necrotic lesions in Late Blight.
- **Attention Coverage Metric:** Calculates the exact percentage of foliar surface under diagnostic activation.

---

## 🌿 Disease Etiology & Classification

| | Class | Scientific Name | Severity | Pathological Signatures |
|---|---|---|---|---|
| 🟤 | **Early Blight** | *Alternaria solani* | Moderate | Dark brown concentric 'target-board' lesions surrounded by chlorotic yellow halos. |
| 🟢 | **Healthy** | — | None | Vibrant green lamina, uniform venation, no necrotic or chlorotic lesions. |
| ⚫ | **Late Blight** | *Phytophthora infestans* | **CRITICAL** | Fast-spreading water-soaked dark necrotic lesions with pale green margins. Destructive pathogen responsible for the Irish Potato Famine. |

---

## ✨ Core Platform Features

- 🧠 **Multi-Model Zoo & Soft Voting:** Select individual paradigms or execute weighted soft-consensus predictions in real time.
- 🔬 **Grad-CAM Saliency Inspector:** Dynamic visual attention heatmap with interactive opacity blending and coverage stats.
- 📊 **Research Benchmark Analytics:** Real-time empirical evaluation metrics dynamically retrieved from training records.
- 🔀 **Test-Time Augmentation (TTA):** Configurable multi-pass geometric averaging (1–15 passes) with temperature scaling ($T = 0.7$).
- 🚨 **Automated Emergency Alert:** Instant warning triggers upon high-confidence detection of *Phytophthora infestans* (Late Blight).
- 💊 **Agronomic Treatment Protocol:** Detailed pathology breakdown including pathogen etiology, chemical fungicides, and cultural prevention.
- ⚡ **Dual Engine Deployment:** Runs full FastAPI + TFLite inference in Docker/local environments with a zero-latency client-side fallback for serverless hosting.
- 🎨 **Academic UI/UX:** Built with Modular Vanilla JS (ES6), SVG iconography, dark/light theme switching, and responsive camera/drag-and-drop inputs.

---

## 📁 Project Architecture

```
potato-leaf-disease-classification/
│
├── backend/                        ← Modular FastAPI REST API Backend
│   ├── __init__.py
│   ├── config.py                   ← Global constants, model zoo metadata & disease info
│   ├── inference.py                ← TFLite interpreter management & TTA engine
│   ├── xai.py                      ← Grad-CAM explainability & tensor saliency generation
│   └── main.py                     ← REST routes & benchmark metrics dependency injection
│
├── model/                          ← Serialized Quantized Model Assets
│   ├── class_names.json            ← Benchmark metrics & class index mappings
│   ├── potato_quantized.tflite     ← Primary quantized inference model
│   └── zoo/                        ← Multi-architecture model zoo (.tflite)
│       ├── densenet121.tflite
│       ├── convnext_tiny.tflite
│       ├── efficientnet_v2s.tflite
│       ├── resnet50.tflite
│       └── mobilenet_v3.tflite
│
├── static/                         ← Modular Vanilla JS Frontend (No Build Step)
│   ├── index.html                  ← Clinical dashboard markup & SVG iconography
│   ├── style.css                   ← CSS design system, glassmorphism & responsive layouts
│   ├── js/
│   │   ├── main.js                 ← Application orchestration & event handling
│   │   ├── api.js                  ← REST API client & error handling
│   │   └── state.js                ← Reactive state management
│   └── samples/                    ← Clinically verified demo leaf images
│
├── kaggle/                         ← Official Kaggle Training Notebooks
│   ├── potato-foliar-pathology-research-benchmark.ipynb
│   └── kernel-metadata.json
│
├── research/                       ← Benchmark & evaluation generation scripts
├── Dockerfile                      ← Container configuration for cloud deployment
├── start.sh                        ← Automated virtual environment & server launcher
├── requirements_app.txt            ← Backend runtime dependencies
├── README.md                       ← Technical documentation
└── LICENSE                         ← MIT License
```

---

## 📊 Dataset Specifications

The model zoo was trained and evaluated on the **Potato Disease Leaf Dataset (PLD)**:
- **Total Images:** 4,072 images (3,251 train / 416 validation / 405 test)
- **Input Resolution:** 256 × 256 × 3 RGB
- **Classes:** Early Blight, Healthy, Late Blight
- **Normalization:** $\frac{\text{RGB}}{255.0}$ floating-point scaling

---

## 💻 Installation & Local Setup

### Quick Start (Recommended)

```bash
# Clone repository
git clone https://github.com/raju-sah/potato-disease-detection-and-classification.git
cd potato-disease-detection-and-classification

# Launch via automated shell script
chmod +x start.sh
./start.sh
```

The script automatically sets up the Python virtual environment, installs dependencies, and starts the Uvicorn ASGI server at `http://localhost:7860`.

### Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements_app.txt
uvicorn backend.main:app --host 0.0.0.0 --port 7860 --reload
```

Interactive OpenAPI Swagger documentation will be accessible at: `http://localhost:7860/docs`.

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status, active models, and input tensor dimensions |
| `GET` | `/api/models` | List all available architectures with empirical accuracy and F1 metrics |
| `GET` | `/api/disease-info` | Full botanical disease encyclopedia, symptoms, and treatment protocols |
| `GET` | `/api/samples` | List verified preset diagnostic sample images |
| `POST` | `/api/predict` | Run multi-model TFLite inference + Grad-CAM saliency mapping |
| `GET` | `/docs` | Interactive Swagger API documentation |

### `POST /api/predict` Parameters

`multipart/form-data`:
- `file` (*required*): Leaf image file (`.jpg`, `.jpeg`, `.png`, `.webp`)
- `model_id` (*optional*, default: `ensemble`): Architecture selection (`ensemble`, `densenet121`, `convnext_tiny`, `efficientnet_v2s`, `resnet50`, `mobilenet_v3`)
- `use_tta` (*optional*, default: `true`): Enable Test-Time Augmentation
- `tta_passes` (*optional*, default: `9`): Number of TTA geometric transformations
- `confidence_threshold` (*optional*, default: `70.0`): Low-confidence warning cutoff

---

## 🛠️ Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Deep Learning & Inference** | TensorFlow 2.15, TensorFlow Lite / LiteRT, Keras, NumPy |
| **Explainable AI (XAI)** | Grad-CAM (Gradient-Weighted Class Activation Mapping) |
| **Backend API** | FastAPI, Uvicorn, Pydantic, Python-Multipart |
| **Frontend Architecture** | Modular Vanilla JavaScript (ES6), HTML5 Canvas, Modern CSS3 |
| **Infrastructure & CI/CD** | Docker, Hugging Face Spaces, GitHub Actions |
| **Research & Benchmarking** | Kaggle GPU Environments, Scikit-Learn, Matplotlib, Seaborn |

---

## 👤 Author & Citations

**Raju Sah**  
- **Email:** [rajucode7@gmail.com](mailto:rajucode7@gmail.com)  
- **Portfolio:** [sahraju.com.np](https://sahraju.com.np)  
- **GitHub:** [@raju-sah](https://github.com/raju-sah)  
- **LinkedIn:** [linkedin.com/in/raju-sah](https://www.linkedin.com/in/raju-sah/)  
- **Kaggle:** [kaggle.com/rajucode](https://www.kaggle.com/rajucode)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<p align="center">
  Made with ❤️ for agricultural AI research and precision plant pathology.
</p>
