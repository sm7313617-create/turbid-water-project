<div align="center">

# TURBID WATER DATASET & OPTICAL DEGRADATION GENERATOR

<p align="center">
  <b>A physically-grounded underwater optical degradation simulator and benchmark dataset for computer vision under severe aquatic turbidity.</b>
</p>

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](generator/app.py)
[![DVC Tracked](https://img.shields.io/badge/DVC-Data_Versioned-9CF0E1?style=for-the-badge&logo=dvc&logoColor=black)](https://dvc.org)
[![QA Verified](https://img.shields.io/badge/QA_Audit-Passed_100%25-10B981?style=for-the-badge&logo=githubactions&logoColor=white)](scripts/audit_fauna_contamination.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<br/>

<table>
  <tr>
    <td align="center"><a href="#overview"><b>Overview</b></a></td>
    <td align="center"><a href="#tech-stack"><b>Tech Stack</b></a></td>
    <td align="center"><a href="#physics-model"><b>Physics Model</b></a></td>
    <td align="center"><a href="#dataset-statistics"><b>Dataset Stats</b></a></td>
    <td align="center"><a href="#installation"><b>Installation</b></a></td>
    <td align="center"><a href="#usage-3-ways"><b>Usage</b></a></td>
    <td align="center"><a href="#data-quality-notes"><b>Data Quality</b></a></td>
  </tr>
</table>

---

</div>

## Tech Stack

<div align="center">

### Core Computation & Math
[![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy_Array_Ops-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy_Math-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![Pandas](https://img.shields.io/badge/Pandas_DataFrames-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

### Vision & Degradation Engine
[![OpenCV](https://img.shields.io/badge/OpenCV_Image_Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Pillow](https://img.shields.io/badge/Pillow_PIL-013243?style=for-the-badge&logo=python&logoColor=white)](https://python-pillow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit_Interactive_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](generator/app.py)

### Data Engineering & Storage
[![DVC](https://img.shields.io/badge/DVC_Data_Version Control-9CF0E1?style=for-the-badge&logo=dvc&logoColor=black)](https://dvc.org)
[![Google Drive](https://img.shields.io/badge/Google_Drive_Remote_Storage-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com)
[![ImageHash](https://img.shields.io/badge/ImageHash_pHash_QA-10B981?style=for-the-badge&logo=hashnode&logoColor=white)](scripts/audit_fauna_contamination.py)
[![COCO Format](https://img.shields.io/badge/COCO_1.0_JSON-00599C?style=for-the-badge&logo=json&logoColor=white)](dataset/dataset_card.md)

</div>

<br/>

| Category | Technologies & Tools | Function & Purpose |
|:---|:---|:---|
| **Scientific Computing** | `Python 3.12` &bull; `NumPy` &bull; `SciPy` &bull; `Pandas` | High-performance array manipulation & physical optical matrix calculations |
| **Computer Vision** | `OpenCV (cv2)` &bull; `Pillow (PIL)` | Spatial image transformations, depth map generation & spatial filtering |
| **Interactive UI** | `Streamlit` | Web-based interactive simulator for real-time optical parameter tuning |
| **Data Engineering** | `DVC` &bull; `Google Drive Remote` | Remote data sync, large file tracking & dataset versioning |
| **Quality Control** | `ImageHash (pHash)` &bull; `tqdm` | Perceptual hash contamination auditing & cluster-aware split verification |
| **Annotations** | `COCO 1.0 Specification` &bull; `PNG Instance Masks` | Polygon segmentations & pixel-level segmentation masks |

---

## Overview

Underwater vision systems face severe performance degradation due to light absorption, backscattering, and suspended sediment. This project provides an end-to-end framework to simulate physical turbidity, verify multi-format annotations, and package a leak-free benchmark synthetic dataset for computer vision model training and evaluation.

> **Project Scope**: Developed for Internship Task 1 of 2.

<details>
<summary><b>Click to view Core Deliverables</b></summary>

* **Degradation Generator**: A physically-grounded optical engine with adjustable turbidity parameters ($0.0$ to $1.0$).
* **Synthetic Labeled Dataset**: 3,384 packaged synthetic images across 4 turbidity levels ($0.2, 0.4, 0.6, 0.8$) and 2 classes (`mangrove_root` and `aquatic_fauna`).
* **Full Data Pipeline**: End-to-end implementation covering collection, annotation, generation, verification, and cluster-aware packaging.

</details>

---

## Physics Model

The degradation engine implements the classic **Jaffe-McGlamery Underwater Image Formation Model**:

$$I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))$$

Where:
* $J(x)$ is the clean input image radiance at scene point $x$.
* $t(x) = \exp(-\beta_c \cdot \text{turbidity} \cdot d(x))$ is the 3-channel transmission map derived from Beer-Lambert's Law, where attenuation coefficients vary by wavelength ($\beta_r = 3.0$, $\beta_g = 1.8$, $\beta_b = 1.0$).
* $d(x)$ is the normalized scene depth map ($0.0 = \text{near}, 1.0 = \text{far}$).
* $A$ is the ambient backscatter vector ($A = [0.10, 0.45, 0.40] \cdot \tau + [0.85, 0.90, 0.95] \cdot (1 - \tau)$).
* $\text{turbidity}$ is a float parameter ranging from $0.0$ (crystal clear) to $1.0$ (maximum turbid).

<details>
<summary><b>Click to view Four Optical Effects</b></summary>

```text
+-----------------------------------------------------------------------------------+
| 1. Wavelength Color Attenuation  --> Selective red light decay (Beer-Lambert Law) |
| 2. Backscatter Haze              --> Additive ambient veil tint (depth-dependent) |
| 3. Forward Scattering Blur       --> Gaussian spatial contrast reduction          |
| 4. Particulate Marine Snow       --> High-frequency suspended sediment noise      |
+-----------------------------------------------------------------------------------+
```

</details>

---

## Dataset Statistics

The packaged dataset contains **3,384 synthetic images** derived from **846 unique base scenes** (482 fauna + 364 mangrove roots) $\times 4$ turbidity levels ($0.2, 0.4, 0.6, 0.8$).

| Split | Class | `turb0.2` | `turb0.4` | `turb0.6` | `turb0.8` | Split Total | Base Scenes |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | `mangrove_root` | 255 | 255 | 255 | 255 | 1,020 | 255 |
| **`train`** | `aquatic_fauna` | 337 | 337 | 337 | 337 | 1,348 | 337 |
| **`train` Subtotal** | | **592** | **592** | **592** | **592** | **2,368 (70%)** | **592** |
| | | | | | | | |
| **`val`** | `mangrove_root` | 54 | 54 | 54 | 54 | 216 | 54 |
| **`val`** | `aquatic_fauna` | 71 | 71 | 71 | 71 | 284 | 71 |
| **`val` Subtotal** | | **125** | **125** | **125** | **125** | **500 (15%)** | **125** |
| | | | | | | | |
| **`test`** | `mangrove_root` | 55 | 55 | 55 | 55 | 220 | 55 |
| **`test`** | `aquatic_fauna` | 74 | 74 | 74 | 74 | 296 | 74 |
| **`test` Subtotal** | | **129** | **129** | **129** | **129** | **516 (15%)** | **129** |
| | | | | | | | |
| **GRAND TOTAL** | **ALL CLASSES** | **846** | **846** | **846** | **846** | **3,384** | **846** |

### Annotation Formats
* **PNG Segmentation Masks**: Used for `aquatic_fauna` base images (pixel-level mask PNGs).
* **COCO JSON Format**: Used for `mangrove_root` images (`train_annotations.coco.json`, `valid_annotations.coco.json`, `test_annotations.coco.json`).

---

## Project Structure

```text
turbid-water-project/
├── data/                         <- DVC-tracked raw data & synthetic outputs (Google Drive)
│   ├── raw/                      <- 482 clean aquatic fauna images (SUIM/DeepFish/F4K)
│   ├── mangrove_frames/          <- 364 annotated mangrove root images (Roboflow export)
│   ├── synthetic/                <- 3,460 turbid synthetic images (generator output)
│   └── annotations/              <- Masks (fauna) & COCO JSON (mangroves)
│       ├── fauna/                <- suim_masks (150), deepfish_masks (182), f4k_masks (150)
│       └── mangrove/             <- train, valid, test COCO annotations
├── dataset/                      <- Final packaged benchmark dataset
│   ├── train/                    <- Train split (2,368 images)
│   ├── val/                      <- Validation split (500 images)
│   ├── test/                     <- Test split (516 images)
│   ├── class_map.json            <- Category mapping {0: mangrove_root, 1: aquatic_fauna}
│   └── dataset_card.md           <- Detailed dataset card & python data loader
├── generator/                    <- Optical Degradation Core Engine
│   ├── app.py                    <- Streamlit UI with turbidity slider
│   ├── degradation.py            <- Core turbidity physics engine
│   ├── generate.py               <- CLI bulk processor
│   └── utils.py                  <- Helper functions
├── labeling/                     <- Annotation verification & frame extraction
│   ├── collectors/               <- Data collection & filter scripts
│   ├── logs/                     <- Selection and skipped CSV logs
│   ├── extract_frames.py         <- Video frame extractor tool
│   └── verify_labels.py          <- Label integrity checker
├── scripts/                      <- QA & Dataset Packaging Utilities
│   ├── audit_fauna_contamination.py <- pHash contamination scanner
│   └── package_dataset.py        <- Dataset splitter with cluster-aware leakage prevention
├── notebooks/                    <- Visual Demos
│   └── 01_degradation_demo.ipynb <- Visual demo notebook of generator
├── docs/                         <- Documentation
│   └── turbidity_model.md        <- Physics equations reference
├── .dvc/                         <- DVC configuration
├── pyrightconfig.json            <- IDE type checker configuration
├── requirements.txt              <- Project dependencies
└── README.md                     <- Project documentation
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/sm7313617-create/turbid-water-project.git
cd turbid-water-project

# 2. Setup virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows PowerShell / CMD

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull DVC-tracked dataset files from Google Drive remote
dvc pull
```

---

## Usage (3 Ways)

### 1. CLI Single Image Generation
Generate a synthetic turbid image with a specific turbidity value:
```bash
python generator/generate.py --input data/raw --output data/synthetic --turbidity 0.5
```

### 2. Bulk Generation (All Turbidity Levels)
Generate all 4 turbidity levels ($0.2, 0.4, 0.6, 0.8$) across an entire directory of clean images:
```bash
python generator/generate.py --input data/raw --output data/synthetic --all-levels
```

### 3. Interactive Streamlit UI
Launch the real-time interactive simulator web app with custom sliders and optical diagnostic charts:
```bash
streamlit run generator/app.py
```

---

## Dataset Integrity

Verify multi-format label integrity and content hash uniqueness across all raw images, masks, and packaged splits:

```bash
python labeling/verify_labels.py
```

All data files are version-controlled using **DVC** and stored on a remote Google Drive storage. Anyone with access can run `dvc pull` to fetch the complete dataset locally without bloating Git history.

---

## Data Quality Notes

1. **Quarantined Contaminated Footage**: An automated perceptual hash audit (`scripts/audit_fauna_contamination.py`) identified 18 initial DeepFish files (`deepfish_00001` through `deepfish_00018`) as mislabeled mangrove underwater dive footage. These 18 stems and their corresponding masks/synthetic variants were quarantined to `_quarantined_mangrove_mislabeled/` folders and excluded from the raw fauna dataset.
2. **Cluster-Aware Leakage Prevention**: Using pHash connected-component clustering ($\text{pHash} \le 8$ bits), 270 near-duplicate video frame clusters were identified across DeepFish and Fish4Knowledge. All images belonging to the same cluster were constrained to land in the exact same split, ensuring **zero data leakage** across train, val, and test splits.
3. **100% Verification Pass**: `verify_labels.py` passed with 0 missing masks across 482 fauna scenes and 364 mangrove frames, with 3,384 unique packaged synthetic image hashes verified.

---

## Data Sources

* **SUIM Dataset**: 150 clean aquatic fauna images and pixel-level segmentation masks.
* **DeepFish Dataset**: 182 clean marine fish images and pixel-level segmentation masks (after quarantining 18 mislabeled stems).
* **Fish4Knowledge (F4K) Dataset**: 150 aquatic fauna images under varying lighting conditions.
* **Underwater Mangrove Footage**: 364 frames extracted from YouTube underwater mangrove dive videos, annotated for `mangrove_root` via Roboflow.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">
  <sub>Developed for the <b>Turbid Water Project</b> &bull; Built with Python, Streamlit, and DVC</sub>
</div>
