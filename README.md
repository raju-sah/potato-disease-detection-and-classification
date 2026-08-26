# 🥔 Potato Leaf Disease Detector
### AI-Powered Plant Disease Classification using EfficientNetB3 + FastAPI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow_Lite-Quantized-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/EfficientNetB3-ImageNet-00BFA5?style=for-the-badge"/>
  <a href="https://www.kaggle.com/code/rajucode/potato-leaf-disease-classification-efficientnetb3">
    <img src="https://img.shields.io/badge/Kaggle_Notebook-Open_in_Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  </a>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  📓 <b>Official Kaggle Training Notebook:</b> <a href="https://www.kaggle.com/code/rajucode/potato-leaf-disease-classification-efficientnetb3"><b>rajucode/potato-leaf-disease-classification-efficientnetb3</b></a>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Kaggle Notebook & Dataset](#-kaggle-notebook--dataset)
- [Disease Classes](#-disease-classes)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation & Local Setup](#-installation--local-setup)
- [API Reference](#-api-reference)
- [Legacy Streamlit App](#-legacy-streamlit-app)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Author](#-author)
- [License](#-license)

---

## 🌿 Overview

This project is an end-to-end **deep learning pipeline** for detecting and classifying **potato leaf diseases** from images. It uses **EfficientNetB3** pretrained on ImageNet and fine-tuned on the **Potato Disease Leaf Dataset (PLD)**. The trained model is converted to **TensorFlow Lite** and served through a **FastAPI backend** with a lightweight **vanilla HTML/CSS/JS frontend**.

> 📓 **Full Training Pipeline:** All data preparation, model training, evaluation, and TFLite quantization can be explored and run in the official Kaggle Notebook:  
> 🔗 **[Kaggle: potato-leaf-disease-classification-efficientnetb3](https://www.kaggle.com/code/rajucode/potato-leaf-disease-classification-efficientnetb3)**

> 🧠 The model was trained on **3,251 images** across 3 classes, achieving high validation accuracy using a two-phase transfer learning approach with label smoothing and test time augmentation.

---

## 🌿 Disease Classes

| | Class | Scientific Name | Severity | Description |
|-|-------|----------------|----------|-------------|
| 🟤 | **Early Blight** | *Alternaria solani* | Moderate | Fungal disease causing dark spots with concentric rings on leaves |
| 🟢 | **Healthy** | — | None | Vibrant green leaves, no spots or damage |
| ⚫ | **Late Blight** | *Phytophthora infestans* | **SEVERE** | Destructive oomycete — caused the Irish Potato Famine. Can devastate crops within days |

---

## ✨ Features

### Web Platform (`server.py`)
- 🔍 **Real-time disease classification** via REST API
- 🎨 **Custom frontend** — drag & drop upload, clipboard paste, one-click sample images
- 📊 **Probability breakdown** for all 3 classes with confidence percentages
- 🔀 **Test Time Augmentation (TTA)** toggle — configurable passes (1–15)
- ⚙️ **Confidence threshold slider** with low-confidence warnings
- 🚨 **Emergency alert** for high-confidence Late Blight detection
- 💊 **Disease diagnostics tabs** — Symptoms, Causes, Treatment, Prevention
- ⚡ **Inference timing** reported per prediction (`inference_time_ms`)
- 🩺 **Health & model info endpoints** for monitoring
- 🌐 **CORS enabled** — call the API from any client

---

## 📁 Project Structure

```
potato-leaf-disease-classification/
│
├── server.py                       ← FastAPI backend + REST API (main app)
├── start.sh                        ← One-command launcher (creates venv, installs deps, starts server)
├── requirements_app.txt            ← Backend dependencies (FastAPI, LiteRT, Pillow…)
├── create_samples.py               ← Generates synthetic demo leaf images
├── packages.txt                    ← System-level packages (Streamlit Cloud, legacy)
├── requirements.txt                ← Streamlit dependencies (legacy)
├── .gitignore
├── README.md
│
├── static/                         ← Frontend (no build step)
│   ├── index.html                  ← UI markup
│   ├── style.css                   ← Styling
│   ├── app.js                      ← Upload, TTA controls, results rendering
│   └── samples/                    ← Demo images served at /static/samples/
│
└── model/
    └── potato_quantized.tflite     ← TFLite quantized model (EfficientNetB3)
```

> `app.py` is the original **Streamlit** app, kept as a legacy alternative (see below).

---

## 🧠 Model Architecture

### Base Model: EfficientNetB3
- Pretrained on **ImageNet** (1000 classes)
- Input size: **256 × 256 × 3**
- Output: 3-class softmax

### Custom Classification Head
```
EfficientNetB3 (frozen backbone)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization → Dropout(0.4)
    ↓
Dense(256, ReLU) + L2 regularization
    ↓
BatchNormalization → Dropout(0.3)
    ↓
Dense(3, Softmax)
```

### Two-Phase Training Strategy

| Phase | Frozen Layers | Learning Rate | Purpose |
|-------|--------------|---------------|---------|
| **Phase 1** — Head only | All EfficientNetB3 | `1e-3` | Learn task-specific features |
| **Phase 2** — Fine-tune | Layers 0→100 | `5e-5` | Adapt deep features to potato domain |

### Key Techniques
| Technique | Details |
|-----------|---------|
| **Label Smoothing** | `ε = 0.1` — prevents overconfident predictions |
| **Class Weights** | Sklearn balanced weights — handles class imbalance |
| **Data Augmentation** | Flip, rotation ±20°, zoom, shift, brightness |
| **TTA** | Averaged augmented passes (mirror/flip/rotations) at inference |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| **Export** | TFLite Float16 Quantized (21 MB, input 256×256×3) |

---

## 📊 Dataset

**Potato Disease Leaf Dataset (PLD)**  
🔗 [https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld)

| Split | Early_Blight | Late_Blight | Healthy | Total |
|-------|-------------|-------------|---------|-------|
| Training | 1,303 | 1,132 | 816 | **3,251** |
| Validation | 163 | 151 | 102 | **416** |
| Testing | 162 | 141 | 102 | **405** |

- Image size: **256 × 256 px**
- Format: `.jpg`
- 3 classes, slight class imbalance (handled via class weights)

---

## 💻 Installation & Local Setup

### Prerequisites
- Python 3.10+
- Bash (Linux / macOS / WSL)

### Quick Start (recommended)

```bash
chmod +x start.sh
./start.sh
```

The script creates a `.venv`, installs `requirements_app.txt`, and starts the server at:

```
http://localhost:8080
```

### Manual Setup

```bash
git clone https://github.com/YOUR_USERNAME/potato-leaf-disease-classification.git
cd potato-leaf-disease-classification

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements_app.txt
python server.py                 # or: uvicorn server:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` in your browser. Interactive API docs at `http://localhost:8080/docs`.

---

## 🔌 API Reference

Base URL: `http://localhost:8080`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check — model status, input shape, class list |
| `GET` | `/api/info` | Model metadata + full disease knowledge base |
| `GET` | `/api/samples` | List available sample images |
| `POST` | `/api/predict` | Classify a leaf image |
| `GET` | `/docs` | Interactive Swagger UI |

### `POST /api/predict`

`multipart/form-data`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Image (JPG / PNG / WEBP) |
| `use_tta` | bool | `false` | Enable Test Time Augmentation |
| `tta_passes` | int | `5` | Number of TTA passes |
| `confidence_threshold` | float | `70.0` | Low-confidence warning cutoff (%) |

```bash
curl -X POST http://localhost:8080/api/predict \
  -F "file=@leaf.jpg" \
  -F "use_tta=true" \
  -F "tta_passes=5"
```

Response (abridged):

```json
{
  "status": "success",
  "prediction": {
    "class_key": "Late_Blight",
    "display_name": "Late Blight",
    "pathogen": "Phytophthora infestans",
    "confidence": 96.42,
    "severity": "CRITICAL ⚠️",
    "urgent_alert": true,
    "is_low_confidence": false
  },
  "probabilities": { "Early_Blight": {...}, "Healthy": {...}, "Late_Blight": {...} },
  "diagnostics": { "symptoms": [...], "causes": [...], "treatment": [...], "prevention": [...] },
  "meta": { "inference_time_ms": 84.31, "tta_applied": true }
}
```

---

## 🖥️ Legacy Streamlit App

The original Streamlit UI (`app.py`) is still included:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Runs at `http://localhost:8501`. Note: `packages.txt` + `requirements.txt` are only needed if deploying that variant to Streamlit Community Cloud.

---

## 📈 Results

Trained and evaluated in the Kaggle notebook `rajucode/potato-leaf-disease-classification-efficientnetb3` (v5, 2026-05). All metrics on the held-out 405-image test set.

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **95.80%** |
| **Macro-F1** | **95.86%** |
| Early Blight F1 | 0.953 |
| Healthy F1 | 0.962 |
| Late Blight F1 | 0.961 |
| Best Val F1 (B3) | 0.967 |
| Ablation — B0 best Val F1 | 0.340 (motivated B3 upgrade) |
| Quantized TFLite Size | ~21 MB (float16) |

> Note: production `model/potato_quantized.tflite` is the EfficientNetB3 checkpoint above (not the older B0-derived weights). Preprocessing is `RGB / 255.0` at 256×256 — matching the notebook.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Core language |
| ![TensorFlow Lite](https://img.shields.io/badge/-TFLite/LiteRT-FF6F00?logo=tensorflow&logoColor=white) | Quantized model inference |
| ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white) | REST API backend |
| ![Uvicorn](https://img.shields.io/badge/-Uvicorn-499588) | ASGI server |
| ![JavaScript](https://img.shields.io/badge/-Vanilla_JS-F7DF1E?logo=javascript&logoColor=black) | Frontend (no framework, no build step) |
| ![Pillow](https://img.shields.io/badge/-Pillow-purple) | Image preprocessing |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Array operations |
| ![Streamlit](https://img.shields.io/badge/-Streamlit_(legacy)-FF4B4B?logo=streamlit&logoColor=white) | Original web app variant |
| ![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?logo=kaggle&logoColor=white) | Training environment & dataset |

---

## 👤 Author

| | |
|-|-|
| **Name** | Raju Sah |
| **Email** | rajucode7@gmail.com |
| **Kaggle Profile** | [kaggle.com/rajucode](https://www.kaggle.com/rajucode) |
| **Kaggle Notebook** | [rajucode/potato-leaf-disease-classification-efficientnetb3](https://www.kaggle.com/code/rajucode/potato-leaf-disease-classification-efficientnetb3) |
| **LinkedIn** | [linkedin.com/in/raju-sah](https://www.linkedin.com/in/raju-sah/) |
| **GitHub** | [github.com/raju-sah](https://github.com/raju-sah) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — free to use, modify, and distribute with attribution.
```

---

## 🙏 Acknowledgements

- Dataset by **[rizwan123456789](https://www.kaggle.com/rizwan123456789)** on Kaggle
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) — Müller et al., 2019
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

---

<p align="center">
  <a href="https://github.com/raju-sah/potato-disease-detection-and-classification/stargazers">
    <img src="https://img.shields.io/github/stars/raju-sah/potato-disease-detection-and-classification?style=social" alt="GitHub Stars"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/raju-sah/potato-disease-detection-and-classification/network/members">
    <img src="https://img.shields.io/github/forks/raju-sah/potato-disease-detection-and-classification?style=social" alt="GitHub Forks"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/raju-sah/potato-disease-detection-and-classification/issues">
    <img src="https://img.shields.io/github/issues/raju-sah/potato-disease-detection-and-classification?style=social" alt="GitHub Issues"/>
  </a>
</p>

<p align="center">
  Made with ❤️ for farmers and agricultural AI research
  <br><br>
  ⭐ <b>Star this repo</b> if it helped you!
  <br><br>
  <a href="#-potato-leaf-disease-detector">⬆️ Back to Top</a>
</p>

<p align="center">
  © 2026 <a href="https://github.com/raju-sah">Raju Sah</a> · All rights reserved under the <a href="LICENSE">MIT License</a>
</p>
