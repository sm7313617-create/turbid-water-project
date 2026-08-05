"""
package_dataset.py
------------------
Packages the turbid water dataset into train/val/test splits for machine learning.

WHY THIS SCRIPT EXISTS
-----------------------
After data collection, label verification, and synthetic turbidity generation,
we need to organize all images and annotations into a clean, standard dataset
structure with reproducible train/val/test splits.

SPLITTING RULES:
1. Base-Image Level Splitting: All 4 turbid variants (0.2, 0.4, 0.6, 0.8) of the
   same base image scene MUST be placed in the same split to prevent data leakage.
2. Mangrove Split Reuse: Reuses pre-existing Roboflow COCO splits (train/valid/test).
3. Fauna Split Stratification: Stratified 70/15/15 random split (seed=42) across
   SUIM, DeepFish, and Fish4Knowledge fauna sub-sources.

DATASET OUTPUT STRUCTURE:
dataset/
  class_map.json
  dataset_card.md
  train/
    images/       (turbid synthetic PNG images for train split)
    annotations/  (PNG masks for fauna + mangrove_annotations.coco.json)
  val/
    images/       (turbid synthetic PNG images for val split)
    annotations/  (PNG masks for fauna + mangrove_annotations.coco.json)
  test/
    images/       (turbid synthetic PNG images for test split)
    annotations/  (PNG masks for fauna + mangrove_annotations.coco.json)
"""

import json
import random
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANGROVE_FRAMES_DIR = DATA_DIR / "mangrove_frames"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

ANNOTATIONS_DIR = DATA_DIR / "annotations"
FAUNA_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "fauna"
MANGROVE_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "mangrove"

DATASET_DIR = PROJECT_ROOT / "dataset"

# Mask subdirectories for fauna sub-sources
FAUNA_MASK_DIRS = {
    "suim": FAUNA_ANNOTATIONS_DIR / "suim_masks",
    "deepfish": FAUNA_ANNOTATIONS_DIR / "deepfish_masks",
    "f4k": FAUNA_ANNOTATIONS_DIR / "f4k_masks",
}

# Source COCO annotation files for mangrove splits
MANGROVE_COCO_FILES = {
    "train": MANGROVE_ANNOTATIONS_DIR / "train_annotations.coco.json",
    "val": MANGROVE_ANNOTATIONS_DIR / "valid_annotations.coco.json",
    "test": MANGROVE_ANNOTATIONS_DIR / "test_annotations.coco.json",
}

TURBIDITY_LEVELS = ["0.2", "0.4", "0.6", "0.8"]
SPLITS = ["train", "val", "test"]


def inspect_synthetic_filenames():
    """
    Prints 5 example filenames from data/synthetic/ to verify the naming pattern
    before executing the packaging logic.
    """
    print("=" * 70)
    print("INSPECTING SYNTHETIC FILENAMES (data/synthetic/)")
    print("=" * 70)
    
    syn_pngs = sorted(list(SYNTHETIC_DIR.glob("*.png")))
    print(f"Total synthetic PNG images found: {len(syn_pngs)}")
    print("Example synthetic filenames:")
    for img_path in syn_pngs[:5]:
        print(f"  - {img_path.name}")
    print()


def prepare_dataset_directories():
    """
    Creates/clears split directories (dataset/<split>/images and dataset/<split>/annotations)
    and removes any existing .gitkeep placeholder files.
    """
    print("Preparing output dataset directory structure...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_dir = DATASET_DIR / split
        
        # Remove old .gitkeep if present
        gitkeep = split_dir / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()
            
        images_dir = split_dir / "images"
        annotations_dir = split_dir / "annotations"

        # Idempotent cleanup: recreate empty folders
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if annotations_dir.exists():
            shutil.rmtree(annotations_dir)

        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)


def get_mangrove_split_assignments():
    """
    Reads the COCO JSON files for mangrove frames to partition base image stems
    into train, val, and test splits.
    
    Returns: dict mapping split -> list of (base_stem, orig_coco_image_entry)
             and dict mapping split -> source COCO data dictionary
    """
    mangrove_stems_by_split = {"train": [], "val": [], "test": []}
    coco_data_by_split = {}

    for split, json_path in MANGROVE_COCO_FILES.items():
        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        coco_data_by_split[split] = coco_data
        
        for img in coco_data.get("images", []):
            file_name = img["file_name"]
            stem = Path(file_name).stem
            mangrove_stems_by_split[split].append((stem, img))

    return mangrove_stems_by_split, coco_data_by_split


def get_fauna_split_assignments(seed=42):
    """
    Collects all fauna base images from data/raw/, groups them by sub-source prefix
    (deepfish, f4k, suim), and performs a stratified 70/15/15 random split.
    
    Returns: dict mapping split -> list of (base_stem, prefix)
    """
    raw_images = sorted([
        f for f in RAW_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    fauna_by_prefix = {}
    for img_path in raw_images:
        stem = img_path.stem
        prefix = stem.split("_")[0].lower()
        fauna_by_prefix.setdefault(prefix, []).append(stem)

    fauna_stems_by_split = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    print("Fauna Stratification Breakdown (70/15/15 split):")
    for prefix, stems in sorted(fauna_by_prefix.items()):
        shuffled = list(stems)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        
        train_stems = shuffled[:n_train]
        val_stems = shuffled[n_train:n_train + n_val]
        test_stems = shuffled[n_train + n_val:]
        
        print(f"  [{prefix:8s}] total: {n:3d} -> train: {len(train_stems):3d}, val: {len(val_stems):3d}, test: {len(test_stems):3d}")
        
        for s in train_stems:
            fauna_stems_by_split["train"].append((s, prefix))
        for s in val_stems:
            fauna_stems_by_split["val"].append((s, prefix))
        for s in test_stems:
            fauna_stems_by_split["test"].append((s, prefix))

    return fauna_stems_by_split


def package_dataset():
    """
    Main packaging routine: copies synthetic images and annotations into dataset/
    and tracks counts for the summary table.
    """
    inspect_synthetic_filenames()
    prepare_dataset_directories()

    mangrove_splits, coco_data_by_split = get_mangrove_split_assignments()
    fauna_splits = get_fauna_split_assignments(seed=42)

    # Pre-index fauna mask files for fast lookup
    fauna_mask_lookup = {}
    for prefix, mask_dir in FAUNA_MASK_DIRS.items():
        if mask_dir.exists():
            for mask_file in mask_dir.glob("*.png"):
                fauna_mask_lookup[mask_file.name] = mask_file

    # Stats tracking structure
    # stats[split][class_name][turb_level] = count
    stats = {
        split: {
            "mangrove_root": {turb: 0 for turb in TURBIDITY_LEVELS},
            "aquatic_fauna": {turb: 0 for turb in TURBIDITY_LEVELS},
        }
        for split in SPLITS
    }

    print("\nProcessing and copying images & annotations...")

    for split in SPLITS:
        split_img_dir = DATASET_DIR / split / "images"
        split_ann_dir = DATASET_DIR / split / "annotations"

        print(f"\n--- Packaging {split.upper()} split ---")

        # -------------------------------------------------------------------
        # 1. MANGROVE PACKAGING
        # -------------------------------------------------------------------
        orig_coco = coco_data_by_split[split]
        new_coco_images = []
        new_coco_annotations = []
        
        # Build lookup for original annotations by image_id
        ann_by_image_id = {}
        for ann in orig_coco.get("annotations", []):
            ann_by_image_id.setdefault(ann["image_id"], []).append(ann)

        img_id_counter = 0
        ann_id_counter = 0

        for stem, img_entry in tqdm(mangrove_splits[split], desc=f"Mangrove ({split})"):
            orig_img_id = img_entry["id"]
            
            for turb in TURBIDITY_LEVELS:
                syn_filename = f"{stem}_turb{turb}.png"
                src_syn_path = SYNTHETIC_DIR / syn_filename
                dst_syn_path = split_img_dir / syn_filename

                if not src_syn_path.exists():
                    print(f"Warning: missing synthetic image {src_syn_path}")
                    continue

                shutil.copy2(src_syn_path, dst_syn_path)
                stats[split]["mangrove_root"][turb] += 1

                # Add COCO image entry
                new_img_id = img_id_counter
                img_id_counter += 1
                
                new_img_entry = dict(img_entry)
                new_img_entry["id"] = new_img_id
                new_img_entry["file_name"] = syn_filename
                new_coco_images.append(new_img_entry)

                # Duplicate annotations for this turbid image variant
                for ann in ann_by_image_id.get(orig_img_id, []):
                    new_ann_entry = dict(ann)
                    new_ann_entry["id"] = ann_id_counter
                    ann_id_counter += 1
                    new_ann_entry["image_id"] = new_img_id
                    new_coco_annotations.append(new_ann_entry)

        # Write filtered/expanded COCO JSON for this split
        new_coco_data = {
            "info": orig_coco.get("info", {}),
            "licenses": orig_coco.get("licenses", []),
            "categories": orig_coco.get("categories", []),
            "images": new_coco_images,
            "annotations": new_coco_annotations,
        }
        out_coco_path = split_ann_dir / "mangrove_annotations.coco.json"
        with open(out_coco_path, "w", encoding="utf-8") as f:
            json.dump(new_coco_data, f, indent=2)

        # -------------------------------------------------------------------
        # 2. FAUNA PACKAGING
        # -------------------------------------------------------------------
        for stem, prefix in tqdm(fauna_splits[split], desc=f"Fauna ({split})"):
            # Copy matching fauna PNG mask
            expected_mask_name = f"{stem}_mask.png"
            if expected_mask_name in fauna_mask_lookup:
                src_mask = fauna_mask_lookup[expected_mask_name]
                dst_mask = split_ann_dir / expected_mask_name
                if not dst_mask.exists():
                    shutil.copy2(src_mask, dst_mask)
            else:
                print(f"Warning: missing mask for fauna stem {stem}")

            # Copy turbid synthetic images
            for turb in TURBIDITY_LEVELS:
                syn_filename = f"{stem}_turb{turb}.png"
                src_syn_path = SYNTHETIC_DIR / syn_filename
                dst_syn_path = split_img_dir / syn_filename

                if not src_syn_path.exists():
                    print(f"Warning: missing synthetic image {src_syn_path}")
                    continue

                shutil.copy2(src_syn_path, dst_syn_path)
                stats[split]["aquatic_fauna"][turb] += 1

    return stats


def print_summary_table(stats):
    """
    Prints a clear, publication-ready Markdown table summarizing dataset counts.
    """
    print("\n" + "=" * 70)
    print("FINAL DATASET PACKAGING SUMMARY")
    print("=" * 70 + "\n")

    header = f"| {'Split':<7} | {'Class':<14} | {'turb0.2':<8} | {'turb0.4':<8} | {'turb0.6':<8} | {'turb0.8':<8} | {'Total':<8} |"
    divider = f"|{'-'*9}|{'-'*16}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|"
    
    print(header)
    print(divider)

    grand_total = 0

    for split in SPLITS:
        split_total = 0
        for cls_name in ["mangrove_root", "aquatic_fauna"]:
            counts = [stats[split][cls_name][t] for t in TURBIDITY_LEVELS]
            cls_total = sum(counts)
            split_total += cls_total
            print(f"| {split:<7} | {cls_name:<14} | {counts[0]:<8d} | {counts[1]:<8d} | {counts[2]:<8d} | {counts[3]:<8d} | {cls_total:<8d} |")
        print(f"| {split:<7} | {'*SUBTOTAL*':<14} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {split_total:<8d} |")
        print(divider)
        grand_total += split_total

    print(f"| {'*TOTAL*':<7} | {'*ALL CLASSES*':<14} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {grand_total:<8d} |")
    print("=" * 70 + "\n")


def main():
    stats = package_dataset()
    print_summary_table(stats)


if __name__ == "__main__":
    main()
