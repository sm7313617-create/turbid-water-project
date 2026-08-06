# Dataset Card: Turbid Water Dataset

## 1. Overview
The **Turbid Water Dataset** is a synthetically-turbidified underwater imagery dataset designed for training, validating, and evaluating computer vision models (object detection and instance segmentation) robust to underwater visibility degradation and turbidity.

The dataset combines clear underwater scenes of **aquatic fauna** (fish, flora, marine life) and **mangrove root systems**, which are synthetically degraded across 4 controlled turbidity levels ($\text{turbidity} \in \{0.2, 0.4, 0.6, 0.8\}$) using a physically-grounded underwater optical propagation model.

---

## 2. Dataset Composition

The dataset contains a total of **3,384 synthetic turbid images** generated from **846 unique base scenes** (482 aquatic fauna scenes and 364 mangrove root scenes). To prevent data leakage during model training and evaluation, all 4 turbid variants of any base scene—along with connected near-duplicate frame clusters—are assigned exclusively to a single split.

### Summary Table

| Split | Class | turb0.2 | turb0.4 | turb0.6 | turb0.8 | Total Images |
|:---|:---|:---|:---|:---|:---|:---|
| **train** | `mangrove_root` | 255 | 255 | 255 | 255 | 1,020 |
| **train** | `aquatic_fauna` | 337 | 337 | 337 | 337 | 1,348 |
| **train** | **SUBTOTAL** | **592** | **592** | **592** | **592** | **2,368** |
| | | | | | | |
| **val** | `mangrove_root` | 54 | 54 | 54 | 54 | 216 |
| **val** | `aquatic_fauna` | 71 | 71 | 71 | 71 | 284 |
| **val** | **SUBTOTAL** | **125** | **125** | **125** | **125** | **500** |
| | | | | | | |
| **test** | `mangrove_root` | 55 | 55 | 55 | 55 | 220 |
| **test** | `aquatic_fauna` | 74 | 74 | 74 | 74 | 296 |
| **test** | **SUBTOTAL** | **129** | **129** | **129** | **129** | **516** |
| | | | | | | |
| **TOTAL**| **ALL CLASSES** | **846** | **846** | **846** | **846** | **3,384** |

---

## 3. Known Data Quality & QA Notes

- **Quarantined Mislabeled Images**: During automated pHash quality assurance, 18 initial DeepFish stems (`deepfish_00001` through `deepfish_00018`) were identified as mislabeled mangrove underwater dive footage rather than genuine marine fish habitats. This mislabeling was confirmed via perceptual hashing (pHash, Hamming distance $\le 8$ bits) and verified with manual visual inspection. These 18 raw image files were quarantined to `data/raw/_quarantined_mangrove_mislabeled/`, their corresponding PNG masks were quarantined to `data/annotations/fauna/deepfish_masks/_quarantined_mangrove_mislabeled/`, and their 72 synthetic turbid variants were moved to `data/synthetic_excluded/_quarantined_mangrove_mislabeled/`.
- **Final Active Fauna & Base Scene Count**: Excluding the 18 quarantined stems brings the active raw fauna image count to **482** (182 DeepFish + 150 SUIM + 150 F4K). Combined with 364 mangrove root scenes, the dataset comprises **846 total base scenes** $\times 4$ turbidity levels = **3,384 packaged synthetic images**.
- **Near-Duplicate Grouping (Leakage Prevention)**: Video-derived public datasets (DeepFish and F4K) contain sequential frames of identical fish/camera angles. Using pHash clustering ($\text{pHash} \le 8$ bits Hamming distance), 270 near-duplicate frame clusters were identified across all fauna sources. Every cluster was constrained to land entirely within a single split (train, val, or test), ensuring **0 clusters span multiple splits** to prevent train-test data leakage.
- **QA Protocol**: Quality control combined automated pHash similarity scanning with manual visual spot-checks across both Type A (mangrove match) and Type B (internal duplicate) flags, as well as across the full Hamming distance range (2–8 bits), confirming all automated flags were accurate and justified.

---

## 4. Annotation Formats

The dataset maintains two complementary annotation formats corresponding to the two underlying data domain types:

1. **Aquatic Fauna (`aquatic_fauna`) — PNG Segmentation Masks**:
   - Stored in `dataset/<split>/annotations/<stem>_mask.png`.
   - Each base fauna image has a pixel-level segmentation mask representing object boundaries.
   - Preserving PNG mask files avoids quantization loss that occurs when converting high-resolution per-pixel segmentation boundaries into sparse COCO polygon approximations.

2. **Mangrove Root (`mangrove_root`) — COCO Format JSON**:
   - Stored in `dataset/<split>/annotations/mangrove_annotations.coco.json`.
   - Contains standard COCO polygon segmentations, bounding boxes, and category IDs.
   - Preserving COCO JSON format preserves multi-instance instance segmentation metadata and bounding box category labels exported from collaborative Roboflow labeling.

---

## 5. How to Load the Dataset in Python

Below is a complete, runnable snippet demonstrating how to load images and their corresponding annotations for any split:

```python
import json
from pathlib import Path
from PIL import Image

# Set dataset base path
DATASET_ROOT = Path("dataset")
SPLIT = "train"  # "train", "val", or "test"

split_dir = DATASET_ROOT / SPLIT
images_dir = split_dir / "images"
annotations_dir = split_dir / "annotations"

# 1. Load a Fauna Image + PNG Mask
fauna_img_path = next(images_dir.glob("*suim*.png"))  # or *deepfish*, *f4k*
base_stem = fauna_img_path.name.split("_turb")[0]
mask_path = annotations_dir / f"{base_stem}_mask.png"

image = Image.open(fauna_img_path)
mask = Image.open(mask_path) if mask_path.exists() else None

print(f"Loaded Fauna Image: {fauna_img_path.name} ({image.size})")
if mask:
    print(f"Loaded Matching Mask: {mask_path.name} ({mask.size})")

# 2. Load Mangrove Image + COCO Annotation Entry
coco_json_path = annotations_dir / "mangrove_annotations.coco.json"
with open(coco_json_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# Find first mangrove image entry
mangrove_img_entry = coco_data["images"][0]
mangrove_img_path = images_dir / mangrove_img_entry["file_name"]
img_id = mangrove_img_entry["id"]

# Retrieve annotations for this image ID
img_annotations = [
    ann for ann in coco_data["annotations"] if ann["image_id"] == img_id
]

print(f"Loaded Mangrove Image: {mangrove_img_path.name}")
print(f"Found {len(img_annotations)} instance annotations for this image.")
```

---

## 6. Turbidity Physics Model

Synthetic turbid images were generated using the **Jaffe-McGlamery Underwater Image Formation Model**, which simulates physical underwater optical degradation according to Beer-Lambert's Law:

$$I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))$$

Where:
- $J(x)$ is the clear input image (radiance).
- $t(x) = \exp(-\beta \cdot d(x))$ is the 3-channel wavelength-dependent transmission map:
  - Red light attenuates rapidly: $\beta_r = 3.0 \times \text{turbidity}$
  - Green light attenuates moderately: $\beta_g = 1.8 \times \text{turbidity}$
  - Blue light penetrates deepest: $\beta_b = 1.0 \times \text{turbidity}$
- $d(x)$ is the estimated depth map of the scene.
- $A$ is the ambient backscatter light vector.
- Additional forward scattering (local contrast reduction / Gaussian blur) and particulate noise (suspended marine snow) are applied as a function of the turbidity parameter.

---

## 7. Data Sources

1. **SUIM (Segmentation of Underwater Imagery)**: 150 clean fauna images and pixel-level masks.
2. **DeepFish**: 182 clean marine fish images and pixel-level masks (after quarantining 18 mislabeled mangrove dive frames).
3. **Fish4Knowledge (F4K)**: 150 aquatic fauna images under varying underwater illumination.
4. **Mangrove YouTube Footage**: 364 frames extracted from underwater mangrove dive videos, annotated for mangrove roots via Roboflow.

---

## 8. Reproduction Steps

To regenerate this dataset from raw source files:

1. **Pull Raw Data**:
   ```bash
   dvc pull
   ```
2. **Audit Fauna Contamination & Near-Duplicates**:
   ```bash
   python scripts/audit_fauna_contamination.py
   ```
3. **Verify Labels & Dataset Integrity**:
   ```bash
   python labeling/verify_labels.py
   ```
4. **Generate Synthetic Turbidity Degradation**:
   ```bash
   python generator/generate.py
   ```
5. **Package Final Dataset Splits**:
   ```bash
   python scripts/package_dataset.py
   ```
