# 🥔 Potato Leaf Disease Detector
### AI-Powered Plant Disease Classification using EfficientNetB3 + Streamlit

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/EfficientNetB3-ImageNet-00BFA5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TFLite-Quantized-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://YOUR_USERNAME-potato-leaf-disease.streamlit.app" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" height="40"/>
  </a>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Disease Classes](#-disease-classes)
- [Demo](#-demo)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation & Local Setup](#-installation--local-setup)
- [Deployment](#-deployment-on-streamlit-community-cloud)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Author](#-author)
- [License](#-license)

---

## 🌿 Overview

This project is an end-to-end **deep learning pipeline** for detecting and classifying **potato leaf diseases** from images. It uses **EfficientNetB3** pretrained on ImageNet and fine-tuned on the **Potato Disease Leaf Dataset (PLD)**. The trained model is converted to **TensorFlow Lite** for fast inference and deployed as a **Streamlit web app**.

> 🧠 The model was trained on **3,251 images** across 3 classes, achieving high validation accuracy using a two-phase transfer learning approach with label smoothing and test time augmentation.

---

## 🌿 Disease Classes

| | Class | Scientific Name | Severity | Description |
|-|-------|----------------|----------|-------------|
| 🟤 | **Early Blight** | *Alternaria solani* | Moderate | Fungal disease causing dark spots with concentric rings on leaves |
| 🟢 | **Healthy** | — | None | Vibrant green leaves, no spots or damage |
| ⚫ | **Late Blight** | *Phytophthora infestans* | **SEVERE** | Destructive oomycete — caused the Irish Potato Famine. Can devastate crops within days |

---

## 🎥 Demo

> 📍 Live App: **[https://YOUR_USERNAME-potato-leaf-disease.streamlit.app](https://YOUR_USERNAME-potato-leaf-disease.streamlit.app)**

**How to use:**
1. Open the app link above
2. Upload a potato leaf image (JPG / PNG)
3. View the predicted disease class, confidence score, and treatment recommendations
4. Optionally enable **TTA** (Test Time Augmentation) from the sidebar for higher accuracy

---

## ✨ Features

- 🔍 **Real-time disease classification** from uploaded leaf images
- 📊 **Probability bar chart** (Plotly) for all 3 classes
- 🔀 **Test Time Augmentation (TTA)** toggle — configurable passes (3–15)
- ⚠️ **Low confidence warning** when prediction is uncertain
- 🚨 **Emergency alert** for high-confidence Late Blight detection
- 💊 **Disease detail tabs** — Symptoms, Causes, Treatment, Prevention
- ⚙️ **Sidebar settings** — TTA toggle, confidence threshold slider
- 📱 Responsive layout — works on desktop and mobile
- ⚡ **TFLite quantized model** — fast inference, ~4× smaller than full model

---

## 📁 Project Structure

```
potato-leaf-disease/
│
├── app.py                        ← Main Streamlit application
├── requirements.txt              ← Python dependencies
├── packages.txt                  ← System-level packages (for Streamlit Cloud)
├── .gitignore
├── README.md
│
└── model/
    └── potato_quantized.tflite   ← TFLite quantized model (EfficientNetB3)
```

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
| **TTA** | 10 augmented passes averaged at inference |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| **Export** | TFLite Float32 + Dynamic Quantized |

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
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/potato-leaf-disease.git
cd potato-leaf-disease
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push your code to **GitHub** (including the `model/` folder)
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Sign in with GitHub → click **"New app"**
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click **"Deploy!"** — done in ~3–5 minutes ✅

> The `packages.txt` file handles system-level dependencies automatically on Streamlit Cloud.

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy (Phase 1) | ~92–94% |
| Best Validation Accuracy (Fine-tune) | ~96–98% |
| TTA Test Accuracy | +1–2% above standard |
| Float32 TFLite Size | ~45 MB |
| Quantized TFLite Size | ~12 MB |
| Compression Ratio | ~4× smaller |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Core language |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | Model training & TFLite export |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Web app framework |
| ![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?logo=plotly&logoColor=white) | Interactive charts |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Array operations |
| ![Pillow](https://img.shields.io/badge/-Pillow-purple) | Image preprocessing |
| ![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikitlearn&logoColor=white) | Class weights & metrics |
| ![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?logo=kaggle&logoColor=white) | Training environment & dataset |

---

## 👤 Author

| | |
|-|-|
| **Name** | Raju Sah |
| **Email** | rajucode7@gmail.com |
| **LinkedIn** | [linkedin.com/in/raju-sah](https://www.linkedin.com/in/raju-sah/) |
| **GitHub** | [github.com/raju-sah](https://github.com/raju-sah) |
| **Kaggle Notebook** | [Potato Leaf Disease — EfficientNetB3](https://www.kaggle.com/) |

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
- [Streamlit Documentation](https://docs.streamlit.io)

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