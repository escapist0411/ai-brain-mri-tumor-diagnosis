<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,40:302b63,100:24243e&height=220&section=header&text=🧠%20Brain%20MRI%20Tumor%20Detection&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=AI-Powered%20Diagnosis%20System%20•%20Deep%20Learning%20•%20Doctor%20Portal&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<br/>

<!-- BADGES ROW 1 -->
<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
&nbsp;
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
&nbsp;
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
&nbsp;
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>

<br/><br/>

<!-- BADGES ROW 2 -->
<img src="https://img.shields.io/badge/Status-Active%20Development-22c55e?style=for-the-badge"/>
&nbsp;
<img src="https://img.shields.io/badge/Type-Academic%20Project-6366f1?style=for-the-badge"/>
&nbsp;
<img src="https://img.shields.io/badge/Runs-100%25%20Offline-0ea5e9?style=for-the-badge"/>
&nbsp;
<img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>

<br/><br/>

> **An end-to-end AI medical diagnosis system** that detects, classifies, and segments brain tumors from MRI scans using deep learning — complete with a professional Doctor Portal and automated PDF diagnostic report generation.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Deep Learning Models](#-deep-learning-models)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Usage Guide](#-usage-guide)
- [Results & Metrics](#-results--metrics)
- [Contributors](#-contributors)

---

## 🔬 Overview

Brain tumor detection and diagnosis is one of the most critical challenges in medical imaging. This project presents a **fully automated AI pipeline** that:

1. 📥 **Accepts** raw MRI brain scans as input
2. 🤖 **Classifies** the tumor type using a trained CNN
3. 🎭 **Augments** training data using DCGAN for rare tumor classes
4. 🗺️ **Segments** the tumor region using U-Net with Dice scoring
5. 📄 **Generates** a professional PDF diagnostic report
6. 🏥 **Presents** everything through a clean Doctor Portal UI

All processing runs **100% offline** on a local machine — no cloud dependency.

---

## 🚀 Key Features

<table>
<tr>
<td width="50%">

### 🧬 AI & Detection
- ✅ **4-class tumor classification**
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- ✅ **Confidence score** per prediction
- ✅ **GAN-based augmentation** for rare classes
- ✅ **U-Net tumor segmentation** with Dice score
- ✅ **Tumor mask overlay** visualization

</td>
<td width="50%">

### 🏥 Doctor Portal
- ✅ Patient registration & management
- ✅ MRI upload & instant diagnosis
- ✅ Real-time visual results dashboard
- ✅ Automated **PDF report generation**
- ✅ **Report history** management
- ✅ Fully offline — runs on local laptop
- ✅ Multi-page Streamlit interface

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DOCTOR PORTAL (Streamlit)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Patient    │  │     MRI      │  │   Report History     │  │
│  │ Registration │  │  Diagnosis   │  │    Management        │  │
│  └──────────────┘  └──────┬───────┘  └──────────────────────┘  │
└─────────────────────────────┼───────────────────────────────────┘
                              │ MRI Input
                ┌─────────────▼─────────────┐
                │        PIPELINE           │
                │                           │
                │  ┌─────────────────────┐  │
                │  │  1. PREPROCESSING   │  │
                │  │  Resize • Normalize │  │
                │  └──────────┬──────────┘  │
                │             │             │
                │  ┌──────────▼──────────┐  │
                │  │   2. CNN CLASSIFIER │  │
                │  │  Glioma / Mening.   │  │
                │  │  Pituitary / None   │  │
                │  └──────────┬──────────┘  │
                │             │             │
                │  ┌──────────▼──────────┐  │
                │  │  3. U-Net SEGMENT.  │  │
                │  │  Tumor Mask Overlay │  │
                │  │  + Dice Score       │  │
                │  └──────────┬──────────┘  │
                │             │             │
                │  ┌──────────▼──────────┐  │
                │  │   4. PDF REPORT     │  │
                │  │  Generate & Save    │  │
                │  └─────────────────────┘  │
                └───────────────────────────┘

                ┌───────────────────────────┐
                │   DCGAN (Background)      │
                │ Synthetic MRI Augmentation│
                │ for Rare Tumor Classes    │
                └───────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

### 🐍 Programming & Frameworks

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### 🧠 Deep Learning & Medical Imaging

![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-0ea5e9?style=for-the-badge&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge&logoColor=white)

### 📄 Reporting

![FPDF](https://img.shields.io/badge/FPDF-e11d48?style=for-the-badge&logoColor=white)
![ReportLab](https://img.shields.io/badge/ReportLab-dc2626?style=for-the-badge&logoColor=white)

</div>

<br/>

| Category | Tool / Library | Purpose |
|----------|---------------|---------|
| **Language** | Python 3.9+ | Core development |
| **DL Framework** | PyTorch | Model training & inference |
| **UI** | Streamlit | Doctor Portal web interface |
| **Classification** | CNN (Custom) | Tumor type classification |
| **Augmentation** | DCGAN | Synthetic MRI generation |
| **Segmentation** | U-Net | Tumor region masking |
| **Image Processing** | OpenCV | MRI preprocessing & overlay |
| **Medical Imaging** | MONAI, Nibabel | DICOM / NIfTI support |
| **Reporting** | FPDF / ReportLab | PDF diagnostic report |
| **Visualization** | Matplotlib, Seaborn | Plots, heatmaps, confusion matrix |

---

## 🧠 Deep Learning Models

### 1️⃣ CNN — Tumor Classifier

```
Input (MRI) → Conv2d → BatchNorm → ReLU → MaxPool
            → Conv2d → BatchNorm → ReLU → MaxPool
            → Flatten → Dense(512) → Dropout
            → Dense(4) → Softmax
            → [Glioma | Meningioma | Pituitary | No Tumor]
```

| Class | Description |
|-------|-------------|
| 🔴 Glioma | Most common malignant brain tumor |
| 🟡 Meningioma | Slow-growing, often benign |
| 🟢 Pituitary | Affects hormonal gland |
| ⚪ No Tumor | Healthy scan — no detection |

---

### 2️⃣ DCGAN — Synthetic MRI Augmentation

Used to generate realistic synthetic MRI images for **underrepresented tumor classes**, improving model generalization and reducing class imbalance.

```
Noise (z) → Generator → Synthetic MRI
Real MRI  → Discriminator → Real / Fake
```

---

### 3️⃣ U-Net — Tumor Segmentation

Encoder-decoder architecture with **skip connections** for precise pixel-level tumor boundary detection.

```
Input → [Encoder: Conv + MaxPool × 4]
      → [Bottleneck]
      → [Decoder: UpConv + Skip Connection × 4]
      → Sigmoid → Binary Tumor Mask
```

**Evaluation Metric:** Dice Similarity Coefficient (DSC)

```
Dice = (2 × |Pred ∩ GT|) / (|Pred| + |GT|)
```

---

## 📂 Project Structure

```
ai-brain-mri-tumor-diagnosis/
│
├── 🚀 app.py                          # Main Doctor Portal entry point
│
├── 📁 pages/                          # Multi-page Streamlit UI
│   ├── 1_Patient_Registration.py      # New patient intake form
│   ├── 2_MRI_Diagnosis.py             # Upload MRI → Run AI → View results
│   ├── 3_Report_Generation.py         # Generate & download PDF report
│   └── 4_Report_History.py            # Browse past reports
│
├── 📁 src/
│   ├── 📁 models/                     # CNN model definition & weights
│   │   ├── cnn_classifier.py
│   │   └── weights/
│   │       └── best_model.pth
│   │
│   ├── 📁 gan/                        # DCGAN for data augmentation
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── train_gan.py
│   │
│   ├── 📁 segmentation/               # U-Net for tumor masking
│   │   ├── unet.py
│   │   ├── train_unet.py
│   │   └── inference.py
│   │
│   ├── 📁 training/                   # Training pipelines
│   │   ├── train_classifier.py
│   │   └── evaluate.py
│   │
│   └── 📁 visualization/              # Overlay, heatmap, charts
│       └── overlay.py
│
├── 📁 data/
│   ├── raw/                           # Original MRI dataset
│   ├── processed/                     # Preprocessed tensors
│   └── synthetic/                     # GAN-generated images
│
├── 📁 reports/
│   ├── figures/                       # Saved plots & overlays
│   └── history/                       # Past PDF diagnostic reports
│
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended for training

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/escapist0411/ai-brain-mri-tumor-diagnosis.git
cd ai-brain-mri-tumor-diagnosis
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

### 3️⃣ Activate the Environment

**Windows (CMD):**
```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Doctor Portal

```bash
streamlit run app.py
```

> 🌐 Opens at **http://localhost:8501** in your browser automatically.

---

## 🖥️ Usage Guide

Once the portal is running:

| Step | Page | Action |
|------|------|--------|
| 1️⃣ | **Patient Registration** | Enter patient name, age, gender, doctor name |
| 2️⃣ | **MRI Diagnosis** | Upload `.jpg` / `.png` MRI scan |
| 3️⃣ | **View Results** | See tumor class, confidence score, segmentation mask |
| 4️⃣ | **Report Generation** | Click "Generate Report" → Download PDF |
| 5️⃣ | **Report History** | Browse & re-download all past reports |

---

## 📊 Results & Metrics

| Model | Metric | Score |
|-------|--------|-------|
| **CNN Classifier** | Accuracy | `~95%` |
| **CNN Classifier** | F1 Score | `~0.94` |
| **U-Net Segmentation** | Dice Score | `~0.87` |
| **DCGAN** | FID Score | *In progress* |

> 📌 Metrics are based on the Brain Tumor MRI Dataset from Kaggle (3,264 images, 4 classes).

---

## 📋 Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
opencv-python>=4.8.0
monai>=1.3.0
nibabel>=5.1.0
fpdf2>=2.7.0
reportlab>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-learn>=1.3.0
```

---

## 🗺️ Roadmap

- [x] CNN tumor classification (4 classes)
- [x] DCGAN synthetic augmentation
- [x] U-Net segmentation with Dice scoring
- [x] Streamlit Doctor Portal (multi-page)
- [x] PDF report generation
- [x] Report history management
- [ ] DICOM file support (.dcm)
- [ ] Grad-CAM explainability heatmaps
- [ ] Multi-patient session support
- [ ] REST API wrapper (FastAPI)
- [ ] Docker containerization

---

## 👥 Contributors

<div align="center">

| <img src="https://avatars.githubusercontent.com/escapist0411" width="80" style="border-radius:50%"/> |
|:---:|
| **Shreyas Sadavarte** |
| [![GitHub](https://img.shields.io/badge/GitHub-escapist0411-181717?style=flat-square&logo=github)](https://github.com/escapist0411) |
| *AI & Deep Learning · Full Stack · Medical Imaging* |

</div>

---

## 📜 License

```
MIT License — Free to use for educational and research purposes.
See LICENSE file for full terms.
```

---

## 🙏 Acknowledgements

- **Dataset:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **U-Net Architecture:** Ronneberger et al. (2015) — *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*
- **MONAI Framework:** Medical Open Network for AI
- **Streamlit** — for making ML apps easy to build

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,40:302b63,100:0f0c29&height=120&section=footer&animation=fadeIn" width="100%"/>

**Built by [Shreyas Sadavarte](https://github.com/escapist0411)**

*If this project helped you, give it a ⭐ on GitHub!*

</div>
