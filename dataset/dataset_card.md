# Dataset Card: Turbid Water Dataset

## 1. Overview
The **Turbid Water Dataset** is a synthetically-turbidified underwater imagery dataset designed for training, validating, and evaluating computer vision models (object detection and instance segmentation) robust to underwater visibility degradation and turbidity.

The dataset combines clear underwater scenes of **aquatic fauna** (fish, flora, marine life) and **mangrove root systems**, which are synthetically degraded across 4 controlled turbidity levels ($\text{turbidity} \in \{0.2, 0.4, 0.6, 0.8\}$) using a physically-grounded underwater optical propagation model.

---

## 2. Dataset Composition

The dataset contains a total of **3,456 synthetic turbid images** generated from **864 unique base scenes** (500 aquatic fauna scenes and 364 mangrove root scenes). To prevent data leakage during model training and evaluation, all 4 turbid variants of any base scene are assigned exclusively to a single split.

### Summary Table

| Split | Class | turb0.2 | turb0.4 | turb0.6 | turb0.8 | Total Images |
|:---|:---|:---|:---|:---|:---|:---|
| **train** | `mangrove_root` | 255 | 255 | 255 | 255 | 1,020 |
| **train** | `aquatic_fauna` | 350 | 350 | 350 | 350 | 1,400 |
| **train** | **SUBTOTAL** | **605** | **605** | **605** | **605** | **2,420** |
| | | | | | | |
| **val** | `mangrove_root` | 54 | 54 | 54 | 54 | 216 |
| **val** | `aquatic_fauna` | 74 | 74 | 74 | 74 | 296 |
| **val** | **SUBTOTAL** | **128** | **128** | **128** | **128** | **512** |
| | | | | | | |
| **test** | `mangrove_root` | 55 | 55 | 55 | 55 | 220 |
| **test** | `aquatic_fauna` | 76 | 76 | 76 | 76 | 304 |
| **test** | **SUBTOTAL** | **131** | **131** | **131** | **131** | **524** |
| | | | | | | |
| **TOTAL**| **ALL CLASSES** | **864** | **864** | **864** | **864** | **3,456** |

---

## 3. Annotation Formats

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

## 4. How to Load the Dataset in Python

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

## 5. Turbidity Physics Model

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

## 6. Data Sources

1. **SUIM (Segmentation of Underwater Imagery)**: 150 clean fauna images and pixel-level masks.
2. **DeepFish**: 200 clean marine fish images and pixel-level masks.
3. **Fish4Knowledge (F4K)**: 150 aquatic fauna images under varying underwater illumination.
4. **Mangrove YouTube Footage**: 364 frames extracted from underwater mangrove dive videos, annotated for mangrove roots via Roboflow.

---

## 7. Reproduction Steps

To regenerate this dataset from raw source files:

1. **Pull Raw Data**:
   ```bash
   dvc pull
   ```
2. **Extract Video Frames** (Optional - reference tool for raw video input):
   ```bash
   python labeling/extract_frames.py <input_video.mp4> <output_folder> --interval 5
   ```
3. **Verify Labels**:
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
