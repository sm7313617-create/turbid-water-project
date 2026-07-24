# Turbid Water Project 🌊

Repository for synthetic turbidity generation, dataset packaging, and annotation verification for underwater mangrove imagery.

## Repository Structure

```text
turbid-water-project/
├── data/                       # All images live here (gitignored)
│   ├── raw/                    # Original clean images from SUIM, DeepFish
│   ├── mangrove_frames/        # Frames extracted from dive videos
│   ├── synthetic/              # Output of your generator (degraded images)
│   └── annotations/            # Labels (.json / mask PNGs)
├── generator/                  # Your main deliverable
│   ├── degradation.py          # Core turbidity physics engine
│   ├── generate.py             # CLI: run on a folder of images
│   ├── app.py                  # Optional Gradio UI with turbidity slider
│   └── utils.py                # Helpers (color shift, noise, blending)
├── labeling/                   # Annotation tools & scripts
│   ├── extract_frames.py       # Pull frames from videos
│   └── verify_labels.py        # Sanity check annotations
├── dataset/                    # Final packaged dataset for Ishan & team
│   ├── train/                  # Split folder: training set
│   ├── val/                    # Split folder: validation set
│   ├── test/                   # Split folder: test set
│   ├── dataset_card.md         # What's in the dataset, how to use
│   └── class_map.json          # {0: mangrove_root, 1: aquatic_fauna}
├── notebooks/                  # Experiments & demos
│   └── 01_degradation_demo.ipynb  # Show your tool working
├── docs/                       # Documentation
│   └── turbidity_model.md      # Physics equations you used
├── README.md
├── requirements.txt
└── .gitignore                  # Ignore data/, *.pyc, .env
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage
- Run CLI generator:
  ```bash
  python generator/generate.py --input data/raw --output data/synthetic --turbidity 0.5
  ```
- Launch Gradio UI:
  ```bash
  python generator/app.py
  ```
