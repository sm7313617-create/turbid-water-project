<div align="center">

<img src="assets/hero_banner.png" alt="Turbid Water Dataset &amp; Optical Degradation Generator" width="100%" />

<br/>

<p align="center">
  <b>Physically-Grounded Underwater Optical Degradation Simulator &amp; Cluster-Aware Benchmark Dataset</b>
</p>

<div align="center">

[![Python](https://img.shields.io/badge/PYTHON-3.12-E63946?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![DVC](https://img.shields.io/badge/DVC-DATA_VERSIONED-1E293B?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org)
[![OpenCV](https://img.shields.io/badge/OPENCV-VISION_ENGINE-E63946?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Model](https://img.shields.io/badge/BEER--LAMBERT-OPTICAL_MODEL-1E293B?style=for-the-badge&logo=scipy&logoColor=white)](docs/turbidity_model.md)

[![Streamlit](https://img.shields.io/badge/STREAMLIT-INTERACTIVE_APP-E63946?style=for-the-badge&logo=streamlit&logoColor=white)](generator/app.py)
[![Images](https://img.shields.io/badge/IMAGES-3%2C384_SYNTHETIC-1E293B?style=for-the-badge&logo=png&logoColor=white)](#dataset-statistics)
[![Levels](https://img.shields.io/badge/TURBIDITY-4_LEVELS-E63946?style=for-the-badge&logo=hashnode&logoColor=white)](#degradation-samples)
[![QA Pass](https://img.shields.io/badge/QA_AUDIT-100%25_PASSED-1E293B?style=for-the-badge&logo=githubactions&logoColor=white)](scripts/audit_fauna_contamination.py)
[![License](https://img.shields.io/badge/LICENSE-MIT-E63946?style=for-the-badge&logo=open-access&logoColor=white)](LICENSE)

</div>

<br/>

<!-- Metric Summary Cards Grid -->
<table>
  <tr>
    <td align="center" width="16.6%">
      <b>DATASET</b><br/>
      <font size="5"><b>846</b></font><br/>
      <sub>Base Scenes</sub><br/>
      <code>482 fauna + 364 mangrove</code>
    </td>
    <td align="center" width="16.6%">
      <b>SYNTHETIC</b><br/>
      <font size="5"><b>3,384</b></font><br/>
      <sub>Packaged Images</sub><br/>
      <code>4 Turbidity Levels</code>
    </td>
    <td align="center" width="16.6%">
      <b>CLASSES</b><br/>
      <font size="5"><b>2</b></font><br/>
      <sub>Target Categories</sub><br/>
      <code>mangrove + fauna</code>
    </td>
    <td align="center" width="16.6%">
      <b>PHYSICS</b><br/>
      <font size="5"><b>Beer-Lambert</b></font><br/>
      <sub>Jaffe-McGlamery</sub><br/>
      <code>4 Degradation Layers</code>
    </td>
    <td align="center" width="16.6%">
      <b>LEAKAGE</b><br/>
      <font size="5"><b>0.0%</b></font><br/>
      <sub>Cross-Split Leakage</sub><br/>
      <code>270 pHash Clusters</code>
    </td>
    <td align="center" width="16.6%">
      <b>VERIFIED</b><br/>
      <font size="5"><b>100%</b></font><br/>
      <sub>Label QA Pass</sub><br/>
      <code>verify_labels.py</code>
    </td>
  </tr>
</table>

</div>

---

> **Semantic segmentation & classification dataset for submerged mangrove root structures and aquatic fauna, paired with a physically grounded Beer-Lambert optical turbidity simulator across 4 controlled degradation levels.**

---

<div align="center">

<p align="center"><b>INDEX &bull; TABLE OF CONTENTS</b></p>

<a href="#overview"><img src="https://img.shields.io/badge/01-OVERVIEW-101b33?style=for-the-badge&labelColor=e63946" alt="01 Overview"/></a>
<a href="#target-classes--annotations"><img src="https://img.shields.io/badge/02-TARGET_CLASSES-101b33?style=for-the-badge&labelColor=e63946" alt="02 Target Classes"/></a>
<a href="#tech-stack"><img src="https://img.shields.io/badge/03-TECH_STACK-101b33?style=for-the-badge&labelColor=e63946" alt="03 Tech Stack"/></a>
<a href="#pipeline-breakdown"><img src="https://img.shields.io/badge/04-PIPELINE_BREAKDOWN-101b33?style=for-the-badge&labelColor=e63946" alt="04 Pipeline Breakdown"/></a>

<a href="#physics-model"><img src="https://img.shields.io/badge/05-PHYSICS_MODEL-101b33?style=for-the-badge&labelColor=e63946" alt="05 Physics Model"/></a>
<a href="#degradation-samples"><img src="https://img.shields.io/badge/06-DEGRADATION_SAMPLES-101b33?style=for-the-badge&labelColor=e63946" alt="06 Degradation Samples"/></a>
<a href="#dataset-statistics"><img src="https://img.shields.io/badge/07-DATASET_STATS-101b33?style=for-the-badge&labelColor=e63946" alt="07 Dataset Stats"/></a>
<a href="#project-structure"><img src="https://img.shields.io/badge/08-STRUCTURE-101b33?style=for-the-badge&labelColor=e63946" alt="08 Structure"/></a>

<a href="#installation"><img src="https://img.shields.io/badge/09-INSTALLATION-101b33?style=for-the-badge&labelColor=e63946" alt="09 Installation"/></a>
<a href="#usage-3-ways"><img src="https://img.shields.io/badge/10-USAGE_GUIDE-101b33?style=for-the-badge&labelColor=e63946" alt="10 Usage Guide"/></a>
<a href="#dataset-integrity"><img src="https://img.shields.io/badge/11-DATA_INTEGRITY-101b33?style=for-the-badge&labelColor=e63946" alt="11 Data Integrity"/></a>
<a href="#data-quality-notes"><img src="https://img.shields.io/badge/12-DATA_QUALITY-101b33?style=for-the-badge&labelColor=e63946" alt="12 Data Quality"/></a>

</div>

---

<a id="overview"></a>
<img src="assets/banners/overview.png" alt="01 // Overview" width="100%" />

<br/>

Underwater vision systems face severe performance degradation due to light absorption, backscattering, and suspended sediment. This project provides an end-to-end framework to simulate physical turbidity, verify multi-format annotations, and package a leak-free benchmark synthetic dataset for computer vision model training and evaluation.

> **Project Scope**: Developed for Internship Task 1 of 2.

<details>
<summary><b>Click to expand Core Deliverables</b></summary>

* **Degradation Generator**: A physically-grounded optical engine with adjustable turbidity parameters ($0.0$ to $1.0$).
* **Synthetic Labeled Dataset**: 3,384 packaged synthetic images across 4 turbidity levels ($0.2, 0.4, 0.6, 0.8$) and 2 classes (`mangrove_root` and `aquatic_fauna`).
* **Full Data Pipeline**: End-to-end implementation covering collection, annotation, generation, verification, and cluster-aware packaging.

</details>

---

<a id="target-classes--annotations"></a>
<img src="assets/banners/classes.png" alt="02 // Target Classes &amp; Annotations" width="100%" />

<br/>

The dataset targets two aquatic computer vision categories, provided with corresponding ground-truth segmentation masks:

| Class 1: `aquatic_fauna` (Raw Image) | Class 1: `aquatic_fauna` (Binary Mask) | Class 2: `mangrove_root` (Raw Image) | Class 2: `mangrove_root` (COCO Mask Overlay) |
|:---:|:---:|:---:|:---:|
| <img src="assets/samples/fauna_clean.jpg" width="220" alt="Aquatic Fauna Raw Image"/> | <img src="assets/samples/fauna_mask.png" width="220" alt="Fauna Mask"/> | <img src="assets/samples/mangrove_clean.jpg" width="220" alt="Mangrove Root Raw Image"/> | <img src="assets/samples/mangrove_annotated.jpg" width="220" alt="Mangrove COCO Overlay"/> |
| *SUIM / DeepFish / F4K Scene* | *Pixel-Level Instance Mask (PNG)* | *Underwater Dive Video Frame* | *COCO 1.0 JSON Polygon Annotation* |

---

<a id="tech-stack"></a>
<img src="assets/banners/techstack.png" alt="03 // Tech Stack" width="100%" />

<br/>

| Domain | Technologies & Badges | Core Function & Implementation |
|:---|:---|:---|
| **Scientific Computing** | [![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org) [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org) [![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org) [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org) | Vectorized RGB matrix attenuation, depth array operations, and dataset metadata management |
| **Computer Vision Engine** | [![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org) [![Pillow](https://img.shields.io/badge/Pillow-013243?style=flat-square&logo=python&logoColor=white)](https://python-pillow.org) | Transmission map filtering, spatial Gaussian contrast reduction, and marine snow synthesis |
| **Interactive UI** | [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](generator/app.py) | Real-time turbidity parameter control, RGB spectral diagnostics, and side-by-side visual comparison |
| **Data Versioning** | [![DVC](https://img.shields.io/badge/DVC-9CF0E1?style=flat-square&logo=dvc&logoColor=black)](https://dvc.org) [![Google Drive](https://img.shields.io/badge/Google_Drive-4285F4?style=flat-square&logo=googledrive&logoColor=white)](https://drive.google.com) | Remote dataset tracking and synchronization without bloating Git version history |
| **Quality Control** | [![ImageHash](https://img.shields.io/badge/ImageHash_pHash-10B981?style=flat-square&logo=hashnode&logoColor=white)](scripts/audit_fauna_contamination.py) [![tqdm](https://img.shields.io/badge/tqdm-FFC107?style=flat-square&logo=python&logoColor=black)](scripts/package_dataset.py) | Perceptual hash contamination auditing ($\text{pHash} \le 8$ bits) & cluster-aware split partitioning |
| **Annotation Standards** | [![COCO 1.0](https://img.shields.io/badge/COCO_1.0_JSON-00599C?style=flat-square&logo=json&logoColor=white)](dataset/dataset_card.md) [![PNG Masks](https://img.shields.io/badge/PNG_Masks-22C55E?style=flat-square&logo=png&logoColor=white)](dataset/dataset_card.md) | Polygon instance segmentations for mangroves and binary pixel-level segmentation masks for fauna |

---

<a id="pipeline-breakdown"></a>
<img src="assets/banners/pipeline.png" alt="04 // Simulation Pipeline Breakdown" width="100%" />

<br/>

The synthetic generator decouples batch orchestration (`generator/generate.py`) from physical optical execution (`generator/degradation.py`).

### End-to-End Execution Flowchart

```mermaid
flowchart LR
    subgraph S1 ["STAGE 01 &bull; BATCH INGESTION & CONTROLS (generate.py)"]
        direction TB
        A["Source Image Folders<br/>(data/raw & data/mangrove_frames)"] --> B["Stem Extraction & Annotation Linkage"]
        B --> C["Turbidity Controller<br/>(0.2, 0.4, 0.6, 0.8)"]
    end

    subgraph S2 ["STAGE 02 &bull; PHYSICAL OPTICS ENGINE (degradation.py)"]
        direction TB
        D["Depth Map Generator d(x)<br/>(Gradient / Radial Profiles)"] --> E["Beer-Lambert Transmission Map t_c(x)<br/>(Wavelength Attenuation: R=3.0, G=1.8, B=1.0)"]
        E --> F["Ambient Backscatter Vector A<br/>(Depth-Weighted Blue-Green Veil)"]
        F --> G["Jaffe-McGlamery Blending<br/>I(x) = J(x) &bull; t(x) + A &bull; (1 - t(x))"]
        G --> H["Scattering Blur & Marine Snow<br/>(Gaussian Contrast Reduction + Particulates)"]
    end

    subgraph S3 ["STAGE 03 &bull; OUTPUT & MANIFEST REGISTRY (generate.py)"]
        direction TB
        I["Traceable Filename<br/>{stem}_turb{level}.png"] --> J["Save Image<br/>(data/synthetic/)"]
        J --> K["Manifest Log Entry<br/>(synthetic_manifest.csv)"]
    end

    S1 --> S2 --> S3

    style S1 fill:#0f172a,stroke:#e63946,stroke-width:2px,color:#f8fafc
    style S2 fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style S3 fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#f8fafc

    style A fill:#1e293b,stroke:#334155,color:#f8fafc
    style B fill:#1e293b,stroke:#334155,color:#f8fafc
    style C fill:#1e293b,stroke:#e63946,color:#f8fafc

    style D fill:#1e293b,stroke:#334155,color:#f8fafc
    style E fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style F fill:#1e293b,stroke:#334155,color:#f8fafc
    style G fill:#1e293b,stroke:#38bdf8,color:#f8fafc
    style H fill:#1e293b,stroke:#334155,color:#f8fafc

    style I fill:#1e293b,stroke:#334155,color:#f8fafc
    style J fill:#1e293b,stroke:#10b981,color:#f8fafc
    style K fill:#1e293b,stroke:#10b981,color:#f8fafc
```

### Component Function Breakdown

| Module File | Component Function | Input Parameters | Output & Purpose |
|:---|:---|:---|:---|
| **`generator/degradation.py`** | `degrade_image()` | Clean RGB Image ($J$), $\text{turbidity} \in [0, 1]$, `--depth-mode` | Orchestrates 6-stage physical optics pipeline and returns degraded RGB image array |
| **`generator/degradation.py`** | `get_per_channel_transmission()` | Depth map $d(x)$, $\text{turbidity}$, $\beta_{r,g,b}$ coefficients | Computes 3-channel Beer-Lambert transmission map $t(x) = \exp(-\beta \cdot \tau \cdot d(x))$ |
| **`generator/generate.py`** | `batch_generate()` | Source dirs (`data/raw`, `data/mangrove_frames`), turbidity levels | Scans clean image directories, executes batch degradation, saves outputs, and writes `synthetic_manifest.csv` |
| **`generator/utils.py`** | Helper Optical Functions | Image array $I$, turbidity level $\tau$, depth map $d(x)$ | Generates normalized depth maps, ambient light vectors, forward scatter blur, and marine snow noise |

---

<a id="physics-model"></a>
<img src="assets/banners/physics.png" alt="05 // Physics Model" width="100%" />

<br/>

The degradation engine implements the classic **Jaffe-McGlamery Underwater Image Formation Model**:

$$I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))$$

Where:
* $J(x)$ is the clean input image radiance at scene point $x$.
* $t(x) = \exp(-\beta_c \cdot \text{turbidity} \cdot d(x))$ is the 3-channel transmission map derived from Beer-Lambert's Law, where attenuation coefficients vary by wavelength ($\beta_r = 3.0$, $\beta_g = 1.8$, $\beta_b = 1.0$).
* $d(x)$ is the normalized scene depth map ($0.0 = \text{near}, 1.0 = \text{far}$).
* $A$ is the ambient backscatter vector ($A = [0.10, 0.45, 0.40] \cdot \tau + [0.85, 0.90, 0.95] \cdot (1 - \tau)$).
* $\text{turbidity}$ is a float parameter ranging from $0.0$ (crystal clear) to $1.0$ (maximum turbid).

<details>
<summary><b>Click to expand Four Optical Degradation Layers</b></summary>

| Effect Layer | Physical Mechanism | Implementation Detail |
|:---|:---|:---|
| **1. Color Attenuation** | Selective red-spectrum absorption | Beer-Lambert exponential decay ($\beta_r = 3.0, \beta_g = 1.8, \beta_b = 1.0$) |
| **2. Backscatter Haze** | Ambient veil accumulation | Depth-weighted additive tinting ($A \cdot (1 - t(x))$) |
| **3. Scattering Blur** | Forward light scattering contrast drop | Dynamic Gaussian spatial kernel ($k = 2 \cdot \lfloor 4\tau \rfloor + 1$) |
| **4. Marine Snow Noise** | Suspended particulate scattering | High-frequency salt-and-pepper noise overlay |

</details>

---

<a id="degradation-samples"></a>
<img src="assets/banners/samples.png" alt="06 // Degradation Samples" width="100%" />

<br/>

### 1. `aquatic_fauna` Degradation Progression

| Base Clear ($\tau = 0.0$) | Turbid $\tau = 0.2$ | Turbid $\tau = 0.4$ | Turbid $\tau = 0.6$ | Turbid $\tau = 0.8$ |
|:---:|:---:|:---:|:---:|:---:|
| <img src="assets/samples/fauna_clean.jpg" width="180" alt="Clean Fauna"/> | <img src="assets/samples/fauna_turb_0.2.jpg" width="180" alt="Fauna Turb 0.2"/> | <img src="assets/samples/fauna_turb_0.4.jpg" width="180" alt="Fauna Turb 0.4"/> | <img src="assets/samples/fauna_turb_0.6.jpg" width="180" alt="Fauna Turb 0.6"/> | <img src="assets/samples/fauna_turb_0.8.jpg" width="180" alt="Fauna Turb 0.8"/> |
| *Clear Radiance $J(x)$* | *Low Attenuation* | *Moderate Haze* | *High Scattering* | *Severe Veil & Noise* |

### 2. `mangrove_root` Degradation Progression

| Base Clear ($\tau = 0.0$) | Turbid $\tau = 0.2$ | Turbid $\tau = 0.4$ | Turbid $\tau = 0.6$ | Turbid $\tau = 0.8$ |
|:---:|:---:|:---:|:---:|:---:|
| <img src="assets/samples/mangrove_clean.jpg" width="180" alt="Clean Mangrove"/> | <img src="assets/samples/mangrove_turb_0.2.jpg" width="180" alt="Mangrove Turb 0.2"/> | <img src="assets/samples/mangrove_turb_0.4.jpg" width="180" alt="Mangrove Turb 0.4"/> | <img src="assets/samples/mangrove_turb_0.6.jpg" width="180" alt="Mangrove Turb 0.6"/> | <img src="assets/samples/mangrove_turb_0.8.jpg" width="180" alt="Mangrove Turb 0.8"/> |
| *Roboflow Frame* | *Low Attenuation* | *Moderate Haze* | *High Scattering* | *Severe Veil & Noise* |

---

<a id="dataset-statistics"></a>
<img src="assets/banners/stats.png" alt="07 // Dataset Statistics" width="100%" />

<br/>

The packaged benchmark dataset contains **3,384 synthetic images** generated from **846 unique base scenes** (482 fauna + 364 mangrove roots) across 4 controlled turbidity levels ($0.2, 0.4, 0.6, 0.8$).

<details open>
<summary><b>Interactive Dataset Breakdown Table</b></summary>

<br/>

| Dataset Split | Target Class | `turb0.2` | `turb0.4` | `turb0.6` | `turb0.8` | Images / Split | Base Scenes | Distribution |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | `mangrove_root` | 255 | 255 | 255 | 255 | 1,020 | 255 | 30.1% |
| **`train`** | `aquatic_fauna` | 337 | 337 | 337 | 337 | 1,348 | 337 | 39.9% |
| **`train` Subtotal** | **All Classes** | **592** | **592** | **592** | **592** | **2,368** | **592** | **70.0%** |
| | | | | | | | | |
| **`val`** | `mangrove_root` | 54 | 54 | 54 | 54 | 216 | 54 | 6.4% |
| **`val`** | `aquatic_fauna` | 71 | 71 | 71 | 71 | 284 | 71 | 8.4% |
| **`val` Subtotal** | **All Classes** | **125** | **125** | **125** | **125** | **500** | **125** | **14.8%** |
| | | | | | | | | |
| **`test`** | `mangrove_root` | 55 | 55 | 55 | 55 | 220 | 55 | 6.5% |
| **`test`** | `aquatic_fauna` | 74 | 74 | 74 | 74 | 296 | 74 | 8.7% |
| **`test` Subtotal** | **All Classes** | **129** | **129** | **129** | **129** | **516** | **129** | **15.2%** |
| | | | | | | | | |
| **BENCHMARK TOTAL** | **`mangrove` + `fauna`** | **846** | **846** | **846** | **846** | **3,384** | **846** | **100.0%** |

</details>

### Annotation Formats
* **PNG Segmentation Masks**: Used for `aquatic_fauna` base images (pixel-level mask PNGs).
* **COCO JSON Format**: Used for `mangrove_root` images (`train_annotations.coco.json`, `valid_annotations.coco.json`, `test_annotations.coco.json`).

---

<a id="project-structure"></a>
<img src="assets/banners/structure.png" alt="08 // Project Structure" width="100%" />

<br/>

```text
turbid-water-project/
├── assets/                       <- Project visual assets, hero banner PNG & section banners
│   ├── hero_banner.png           <- Japanese minimalist design hero banner
│   ├── banners/                  <- Japanese aesthetic section header banners
│   └── samples/                  <- Raw images, mask samples & degradation progressions
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

<a id="installation"></a>
<img src="assets/banners/installation.png" alt="09 // Installation" width="100%" />

<br/>

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

<a id="usage-3-ways"></a>
<img src="assets/banners/usage.png" alt="10 // Usage Guide" width="100%" />

<br/>

| Execution Mode | Primary Purpose | Command |
|:---|:---|:---|
| **1. CLI Single Image** | Quick test on a single image with custom turbidity value | `python generator/generate.py --input data/raw --output data/synthetic --turbidity 0.5` |
| **2. Bulk Directory** | Generate all 4 turbidity levels ($0.2, 0.4, 0.6, 0.8$) in bulk | `python generator/generate.py --input data/raw --output data/synthetic --all-levels` |
| **3. Streamlit Web App** | Interactive parameter exploration, real-time preview & metrics | `streamlit run generator/app.py` |

---

<a id="dataset-integrity"></a>
<img src="assets/banners/integrity.png" alt="11 // Dataset Integrity" width="100%" />

<br/>

Verify multi-format label integrity and content hash uniqueness across all raw images, masks, and packaged splits:

```bash
python labeling/verify_labels.py
```

All data files are version-controlled using **DVC** and stored on a remote Google Drive storage. Anyone with access can run `dvc pull` to fetch the complete dataset locally without bloating Git history.

---

<a id="data-quality-notes"></a>
<img src="assets/banners/quality.png" alt="12 // Data Quality Notes" width="100%" />

<br/>

1. **Quarantined Contaminated Footage**: An automated perceptual hash audit (`scripts/audit_fauna_contamination.py`) identified 18 initial DeepFish files (`deepfish_00001` through `deepfish_00018`) as mislabeled mangrove underwater dive footage. These 18 stems and their corresponding masks/synthetic variants were quarantined to `_quarantined_mangrove_mislabeled/` folders and excluded from the raw fauna dataset.
2. **Cluster-Aware Leakage Prevention**: Using pHash connected-component clustering ($\text{pHash} \le 8$ bits), 270 near-duplicate video frame clusters were identified across DeepFish and Fish4Knowledge. All images belonging to the same cluster were constrained to land in the exact same split, ensuring **zero data leakage** across train, val, and test splits.
3. **100% Verification Pass**: `verify_labels.py` passed with 0 missing masks across 482 fauna scenes and 364 mangrove frames, with 3,384 unique packaged synthetic image hashes verified.

---

<a id="data-sources"></a>
<img src="assets/banners/sources.png" alt="13 // Data Sources" width="100%" />

<br/>

* **SUIM Dataset**: 150 clean aquatic fauna images and pixel-level segmentation masks.
* **DeepFish Dataset**: 182 clean marine fish images and pixel-level segmentation masks (after quarantining 18 mislabeled stems).
* **Fish4Knowledge (F4K) Dataset**: 150 aquatic fauna images under varying lighting conditions.
* **Underwater Mangrove Footage**: 364 frames extracted from YouTube underwater mangrove dive videos, annotated for `mangrove_root` via Roboflow.

---

<a id="license"></a>
<img src="assets/banners/license.png" alt="14 // License" width="100%" />

<br/>

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">
  <sub>Developed for the <b>Turbid Water Project</b> &bull; Built with Python, Streamlit, and DVC</sub>
</div>
