# Turbid Water Dataset & Degradation Generator

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![DVC Tracked](https://img.shields.io/badge/DVC-Data_Versioned-9CF0E1?style=for-the-badge&logo=dvc&logoColor=black)](https://dvc.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end framework that applies physically-grounded underwater optical degradation to clean imagery, producing a benchmark synthetic turbid dataset covering two target classes: `mangrove_root` and `aquatic_fauna`. Developed for Internship Task 1 of 2.

---

## Overview

Underwater vision systems face severe degradation due to light absorption, backscattering, and suspended sediment. This project provides a complete pipeline to generate, verify, and package a synthetic turbid dataset to train and evaluate computer vision models robust to visibility loss.

### Deliverables
* **Degradation Generator**: A physically-grounded optical engine with adjustable turbidity parameters ($0.0$ to $1.0$).
* **Synthetic Labeled Dataset**: 3,384 packaged synthetic images across 4 turbidity levels ($0.2, 0.4, 0.6, 0.8$) and 2 classes (`mangrove_root` and `aquatic_fauna`).
* **Full Data Pipeline**: End-to-end implementation covering collection, annotation, generation, verification, and cluster-aware packaging.

---

## Tech Stack

* **Core Language & Scientific Computing**: Python 3.12, NumPy, SciPy, Pandas
* **Computer Vision & Image Processing**: OpenCV (`cv2`), Pillow (`PIL`)
* **Interactive Simulator Interface**: Streamlit
* **Data Versioning & Remote Storage**: DVC (Data Version Control), Google Drive Remote (`dvc-gdrive`)
* **Quality Assurance & Perceptual Hashing**: ImageHash (pHash algorithm), `tqdm`
* **Data Annotation Specifications**: COCO 1.0 JSON Format, Binary PNG Instance Masks

---

## Physics Model

The generator simulates optical degradation using the **Jaffe-McGlamery Underwater Image Formation Model**:

$$I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))$$

Where:
* $J(x)$ is the clean input image radiance.
* $t(x) = \exp(-\beta_c \cdot \text{turbidity} \cdot d(x))$ is the 3-channel transmission map derived from Beer-Lambert's Law, where attenuation coefficients vary by wavelength ($\beta_r = 3.0$, $\beta_g = 1.8$, $\beta_b = 1.0$).
* $d(x)$ is the normalized scene depth map ($0.0 = \text{near}, 1.0 = \text{far}$).
* $A$ is the ambient backscatter vector (blue-green tinted veil).
* $\text{turbidity}$ is a float parameter ranging from $0.0$ (crystal clear) to $1.0$ (maximum turbid).

### Four Degradation Effects Applied
1. **Wavelength Color Attenuation**: Selective absorption causing red light to decay faster than green/blue.
2. **Backscatter Haze**: Additive ambient veil increasing with distance and turbidity.
3. **Forward Scattering Blur**: Local spatial contrast reduction via Gaussian kernel scaling.
4. **Particulate Marine Snow**: Random high-frequency particle noise representing suspended sediment.

---

## Dataset Statistics

The packaged dataset contains **3,384 synthetic images** derived from **846 unique base scenes** (482 fauna + 364 mangrove roots) $\times 4$ turbidity levels ($0.2, 0.4, 0.6, 0.8$).

| Split | Class | turb0.2 | turb0.4 | turb0.6 | turb0.8 | Subtotal | Base Scenes |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **train** | `mangrove_root` | 255 | 255 | 255 | 255 | 1,020 | 255 |
| **train** | `aquatic_fauna` | 337 | 337 | 337 | 337 | 1,348 | 337 |
| **train Subtotal** | | **592** | **592** | **592** | **592** | **2,368 (70%)** | **592** |
| | | | | | | | |
| **val** | `mangrove_root` | 54 | 54 | 54 | 54 | 216 | 54 |
| **val** | `aquatic_fauna` | 71 | 71 | 71 | 71 | 284 | 71 |
| **val Subtotal** | | **125** | **125** | **125** | **125** | **500 (15%)** | **125** |
| | | | | | | | |
| **test** | `mangrove_root` | 55 | 55 | 55 | 55 | 220 | 55 |
| **test** | `aquatic_fauna` | 74 | 74 | 74 | 74 | 296 | 74 |
| **test Subtotal** | | **129** | **129** | **129** | **129** | **516 (15%)** | **129** |
| | | | | | | | |
| **TOTAL** | **ALL CLASSES** | **846** | **846** | **846** | **846** | **3,384** | **846** |

### Annotation Formats
* **PNG Segmentation Masks**: Used for `aquatic_fauna` base images (pixel-level mask PNGs).
* **COCO JSON Format**: Used for `mangrove_root` images (`train_annotations.coco.json`, `valid_annotations.coco.json`, `test_annotations.coco.json`).

---

## Project Structure

```text
turbid-water-project/
  data/
    raw/                          ← 482 clean aquatic fauna images (SUIM/DeepFish/F4K)
    mangrove_frames/              ← 364 annotated mangrove root images (Roboflow export)
    synthetic/                    ← 3,460 turbid synthetic images (generator output)
    annotations/
      fauna/
        suim_masks/               ← 150 PNG masks
        deepfish_masks/           ← 182 PNG masks (18 quarantined, contamination caught)
        f4k_masks/                ← 150 PNG masks
      mangrove/
        train_annotations.coco.json
        valid_annotations.coco.json
        test_annotations.coco.json
  dataset/
    train/                        ← 2,368 images (70%)
    val/                          ← 500 images (15%)
    test/                         ← 516 images (15%)
    class_map.json
    dataset_card.md
  generator/
    degradation.py                ← core turbidity physics engine
    utils.py                      ← helper functions
    generate.py                   ← CLI bulk processor
    app.py                        ← Streamlit UI with turbidity slider
  labeling/
    collectors/                   ← filter_suim.py, filter_deepfish.py, filter_fish4knowledge.py
    logs/                         ← selection and skipped CSVs
    extract_frames.py             ← video frame extractor tool
    verify_labels.py              ← label integrity checker
  scripts/
    package_dataset.py            ← dataset splitter with cluster-aware leakage prevention
    audit_fauna_contamination.py  ← pHash contamination scanner
  notebooks/
    01_degradation_demo.ipynb     ← visual demo of the generator
  docs/
    turbidity_model.md
  .dvc/                           ← DVC configuration
  requirements.txt
  .gitignore
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sm7313617-create/turbid-water-project.git
cd turbid-water-project

# Setup virtual environment
python -m venv venv
venv\Scripts\activate        # Windows PowerShell / CMD

# Install dependencies
pip install -r requirements.txt

# Pull DVC-tracked dataset files from Google Drive remote
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

1. **SUIM Dataset**: 150 clean aquatic fauna images and pixel-level segmentation masks.
2. **DeepFish Dataset**: 182 clean marine fish images and pixel-level segmentation masks (after quarantining 18 mislabeled stems).
3. **Fish4Knowledge (F4K) Dataset**: 150 aquatic fauna images under varying lighting conditions.
4. **Underwater Mangrove Footage**: 364 frames extracted from YouTube underwater mangrove dive videos, annotated for `mangrove_root` via Roboflow.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
