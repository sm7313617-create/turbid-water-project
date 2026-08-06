<div align="center">

# Turbid Water Dataset & Optical Degradation Generator

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](generator/app.py)
[![DVC Tracked](https://img.shields.io/badge/DVC-Data_Versioned-9CF0E1?style=for-the-badge&logo=dvc&logoColor=black)](https://dvc.org)
[![QA Verified](https://img.shields.io/badge/QA_Audit-Passed_100%25-10B981?style=for-the-badge&logo=githubactions&logoColor=white)](scripts/audit_fauna_contamination.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A physically-grounded underwater optical degradation simulator and benchmark dataset for computer vision under severe aquatic turbidity.</b>
</p>

[Key Features](#key-features) •
[Interactive Web App](#interactive-simulator-app) •
[Physics Model](#optical-physics-model) •
[Dataset Benchmark](#dataset-composition--benchmark) •
[Quickstart](#quick-start) •
[Quality Assurance](#quality-assurance--leakage-prevention)

---

</div>

## Executive Summary

Computer vision models deployed in underwater environments (such as autonomous underwater vehicles, marine ecology monitoring, and mangrove root inspection) suffer severe performance degradation due to **light attenuation, backscatter, and visibility loss**.

The **Turbid Water Project** provides an end-to-end framework:
1. **Physically Grounded Degradation Engine**: Simulates realistic underwater optical degradation using the **Jaffe-McGlamery Optical Formation Model** and **Beer-Lambert's Law**.
2. **Interactive Streamlit Web Simulator**: A clean, professional Web UI for real-time parameter tuning, visual comparison, and spectral transmission diagnostics.
3. **3,384 Synthetic Turbid Benchmark Dataset**: 846 base scenes (482 aquatic fauna + 364 mangrove roots) synthetically degraded across 4 controlled turbidity levels ($\text{turbidity} \in \{0.2, 0.4, 0.6, 0.8\}$).
4. **Perceptual Hash QA & Cluster-Aware Splitting**: Automated pHash contamination audit ($\text{pHash} \le 8$ bits) preventing train-test data leakage across sequential video frames.

---

## Key Features

* **Physically Accurately Wavelength Attenuation**: Simulates exponential light decay where red wavelengths ($\lambda \approx 700\text{ nm}$) attenuate rapidly ($\beta_r = 3.0 \cdot \tau$), green attenuates moderately ($\beta_g = 1.8 \cdot \tau$), and blue penetrates deepest ($\beta_b = 1.0 \cdot \tau$).
* **Ambient Backscatter & Marine Snow**: Generates depth-dependent ambient veil tint and forward-scattering particle noise.
* **Perceptual Contamination Audit**: Custom pHash detection engine ([scripts/audit_fauna_contamination.py](file:///c:/DTE%204%20Calamity%20and%20Humanity%20Pvt%20Ltd%20%28DTECH%29/turbid-water-project/scripts/audit_fauna_contamination.py)) to detect near-duplicate frame clusters and mislabeled footage.
* **Leakage-Free Dataset Packaging**: 270 near-duplicate video frame clusters constrained to single splits (**0 leakage across splits**).
* **Dual Format Support**: COCO 1.0 JSON polygon segmentations for `mangrove_root` + PNG segmentation masks for `aquatic_fauna`.

---

## Interactive Simulator App

The repository includes a web interface built with **Streamlit** for visual parameter exploration:

```bash
# Launch the interactive web simulator
streamlit run generator/app.py
```

### App Highlights
* **Custom Gradient Turbidity Slider**: Adjust turbidity $\tau \in [0.0, 1.0]$ in real-time.
* **Depth Map Profiles**: Toggle between **linear vertical gradient** (perspective depth) and **radial distance** (center subject focus).
* **Spectral Attenuation Metrics**: Live RGB transmission range metrics and channel degradation bar charts.
* **Dataset Explorer & Uploads**: Select clean fauna/mangrove scenes directly from raw data or upload custom images.

---

## Optical Physics Model

The degradation engine implements the classic **Jaffe-McGlamery Underwater Image Formation Model**:

$$I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))$$

Where:
* **$J(x)$**: Clear input image radiance at scene point $x$.
* **$t(x) \in [0, 1]$**: 3-channel wavelength-dependent transmission map defined by Beer-Lambert's Law:
  $$t_c(x) = \exp\left(-\beta_c \cdot \text{turbidity} \cdot d(x)\right), \quad c \in \{R, G, B\}$$
* **$d(x)$**: Normalized depth map of the scene ($0.0 = \text{near}, 1.0 = \text{far}$).
* **$A$**: Atmospheric/ambient backscatter light vector ($A = [0.10, 0.45, 0.40] \cdot \tau + [0.85, 0.90, 0.95] \cdot (1 - \tau)$).
* **Forward Scattering**: High-frequency contrast reduction via Gaussian spatial blur ($k = 2 \cdot \lfloor 4 \tau \rfloor + 1$).

---

## Dataset Composition & Benchmark

The dataset is partitioned into `train`, `val`, and `test` splits. All 4 turbid variants of any base scene—along with connected near-duplicate frame clusters—are kept strictly within the same split.

| Split | Class | `turb0.2` | `turb0.4` | `turb0.6` | `turb0.8` | Split Total | Base Scenes |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | `mangrove_root` | 255 | 255 | 255 | 255 | 1,020 | 255 |
| **`train`** | `aquatic_fauna` | 337 | 337 | 337 | 337 | 1,348 | 337 |
| **`train` Subtotal** | | **592** | **592** | **592** | **592** | **2,368** | **592** |
| | | | | | | | |
| **`val`** | `mangrove_root` | 54 | 54 | 54 | 54 | 216 | 54 |
| **`val`** | `aquatic_fauna` | 71 | 71 | 71 | 71 | 284 | 71 |
| **`val` Subtotal** | | **125** | **125** | **125** | **125** | **500** | **125** |
| | | | | | | | |
| **`test`** | `mangrove_root` | 55 | 55 | 55 | 55 | 220 | 55 |
| **`test`** | `aquatic_fauna` | 74 | 74 | 74 | 74 | 296 | 74 |
| **`test` Subtotal** | | **129** | **129** | **129** | **129** | **516** | **129** |
| | | | | | | | |
| **GRAND TOTAL** | **ALL CLASSES** | **846** | **846** | **846** | **846** | **3,384** | **846** |

---

## Quality Assurance & Leakage Prevention

<details>
<summary><b>Click to expand Quality Control Details</b></summary>

### 1. Quarantined Mislabeled Footage
During automated perceptual hashing QA ([scripts/audit_fauna_contamination.py](file:///c:/DTE%204%20Calamity%20and%20Humanity%20Pvt%20Ltd%20%28DTECH%29/turbid-water-project/scripts/audit_fauna_contamination.py)), 18 initial DeepFish files (`deepfish_00001` through `deepfish_00018`) were flagged as mislabeled underwater mangrove dive footage matching `data/mangrove_frames/`. These 18 stems, their PNG masks, and 72 synthetic variants were quarantined into `_quarantined_mangrove_mislabeled/` subdirectories and excluded from the raw benchmark.

### 2. Cluster-Aware Leakage Prevention
To eliminate train-test data leakage caused by sequential video frame redundancy in public video datasets (DeepFish and Fish4Knowledge), pHash connected-component clustering ($\text{pHash} \le 8$ bits Hamming distance) identified **270 near-duplicate clusters**. Every cluster is assigned exclusively to a single split (**0 leakage across splits**).

</details>

---

## Quick Start

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/sm7313617-create/turbid-water-project.git
cd turbid-water-project

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows PowerShell
```

### 2. Install Dependencies & Fetch Data
```bash
pip install -r requirements.txt

# Pull DVC-tracked raw data and dataset splits from Google Drive
dvc pull
```

### 3. Run Synthetic Batch Generator
```bash
# Generate synthetic turbid images for raw fauna
python generator/generate.py --input data/raw --output data/synthetic --turbidity 0.4
```

### 4. Verify Annotations & Package Dataset
```bash
# Run multi-format label verification
python labeling/verify_labels.py

# Package dataset into train/val/test splits
python scripts/package_dataset.py
```

---

## Repository Architecture

```text
turbid-water-project/
├── data/                       # DVC-tracked raw data & synthetic outputs (Google Drive)
│   ├── raw/                    # Clean fauna images (182 DeepFish, 150 SUIM, 150 F4K = 482)
│   ├── mangrove_frames/        # 364 extracted mangrove dive frames
│   ├── synthetic/              # 3,384 generated turbid images
│   └── annotations/            # Masks (fauna) & COCO JSON (mangroves)
├── dataset/                    # Final packaged benchmark dataset
│   ├── train/                  # Train split (2,368 images)
│   ├── val/                    # Validation split (500 images)
│   ├── test/                   # Test split (516 images)
│   ├── dataset_card.md         # Detailed dataset card & python data loader
│   └── class_map.json          # Category mapping {0: mangrove_root, 1: aquatic_fauna}
├── generator/                  # Optical Degradation Core Engine
│   ├── app.py                  # Interactive Streamlit Web Simulator App
│   ├── degradation.py          # Physics engine (transmission maps, backscatter, blur)
│   ├── generate.py             # Batch CLI generator script
│   └── utils.py                # Image IO, depth map generation, particle noise
├── labeling/                   # Annotation verification & frame extraction
│   ├── extract_frames.py       # Frame extraction utility from YouTube video logs
│   └── verify_labels.py        # Multi-format annotation integrity & hash uniqueness verifier
├── scripts/                    # QA & Dataset Packaging Utilities
│   ├── audit_fauna_contamination.py  # pHash contamination & near-duplicate cluster audit
│   └── package_dataset.py      # Cluster-aware dataset packaging script
├── pyrightconfig.json          # IDE type checker configuration
├── requirements.txt            # Project Python dependencies
└── README.md                   # Project documentation
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">
  <sub>Developed for the <b>Turbid Water Project</b> • Built with Python, Streamlit, and DVC</sub>
</div>
