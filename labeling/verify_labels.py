"""
verify_labels.py
-----------------
Checks that every image in the project has a matching annotation.

This project has TWO different annotation formats, so this script checks
both, separately, and then prints one combined final verdict.

1) FAUNA IMAGES (data/raw/) -> PNG segmentation masks
   ---------------------------------------------------
   Masks live in three folders:
       data/annotations/fauna/suim_masks/
       data/annotations/fauna/deepfish_masks/
       data/annotations/fauna/f4k_masks/

   Matching rule: an image "suim_00001.jpg" must have a mask file called
   "suim_00001_mask.png" somewhere in one of the three mask folders.

2) MANGROVE IMAGES (data/mangrove_frames/) -> COCO JSON annotations
   ------------------------------------------------------------------
   Three JSON files live in data/annotations/mangrove/:
       train_annotations.coco.json
       valid_annotations.coco.json
       test_annotations.coco.json

   Each COCO JSON has an "images" list, and each entry has a "file_name"
   field. We check two directions:
       a) Every image file on disk appears in at least one JSON's
          "images" list (i.e. it has been labeled/assigned).
       b) Every file_name mentioned inside the JSONs actually exists as
          a real file in data/mangrove_frames/ (i.e. no "ghost" entries
          pointing at missing files).

HOW TO RUN
----------
    python labeling/verify_labels.py

(Run it from the project root folder, turbid-water-project/, so the
relative data/ paths resolve correctly. It also works run from inside
labeling/ - see PROJECT_ROOT below.)
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass



# ---------------------------------------------------------------------------
# SET UP PATHS
# ---------------------------------------------------------------------------
# __file__ is this script's own location: .../turbid-water-project/labeling/verify_labels.py
# .parent            -> .../turbid-water-project/labeling
# .parent.parent     -> .../turbid-water-project        (the project root)
# Using this trick means the script works no matter which folder you run it FROM.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw"
MANGROVE_IMAGES_DIR = DATA_DIR / "mangrove_frames"

FAUNA_MASK_DIRS = [
    DATA_DIR / "annotations" / "fauna" / "suim_masks",
    DATA_DIR / "annotations" / "fauna" / "deepfish_masks",
    DATA_DIR / "annotations" / "fauna" / "f4k_masks",
]

MANGROVE_ANNOTATION_DIR = DATA_DIR / "annotations" / "mangrove"
MANGROVE_JSON_FILES = [
    MANGROVE_ANNOTATION_DIR / "train_annotations.coco.json",
    MANGROVE_ANNOTATION_DIR / "valid_annotations.coco.json",
    MANGROVE_ANNOTATION_DIR / "test_annotations.coco.json",
]

# Which file extensions count as "images" when scanning folders
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# PART 1: FAUNA CHECK (PNG masks)
# ---------------------------------------------------------------------------
def check_fauna_labels():
    """
    Returns (passed: bool, missing_masks: list[str], total_images: int)
    """
    print("=" * 70)
    print("CHECKING FAUNA IMAGES (data/raw/) against PNG masks")
    print("=" * 70)

    if not RAW_IMAGES_DIR.exists():
        print(f"ERROR: folder not found: {RAW_IMAGES_DIR}")
        return False, [], 0

    # Collect all image files in data/raw/
    raw_images = sorted(
        f for f in RAW_IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    # Build a lookup set of every mask filename that exists, across all
    # three mask folders. A set makes "does this mask exist?" checks fast.
    existing_mask_names = set()
    for mask_dir in FAUNA_MASK_DIRS:
        if not mask_dir.exists():
            print(f"Warning: mask folder not found: {mask_dir}")
            continue
        for mask_file in mask_dir.iterdir():
            if mask_file.is_file() and mask_file.suffix.lower() == ".png":
                existing_mask_names.add(mask_file.name)

    missing_masks = []

    for image_path in raw_images:
        # "suim_00001.jpg" -> stem is "suim_00001"
        stem = image_path.stem
        expected_mask_name = f"{stem}_mask.png"

        if expected_mask_name not in existing_mask_names:
            missing_masks.append(f"{image_path.name}  ->  missing {expected_mask_name}")

    total_images = len(raw_images)
    passed = len(missing_masks) == 0

    # --- Print results ---
    print(f"Images found in data/raw/:        {total_images}")
    print(f"Mask files found (all 3 folders): {len(existing_mask_names)}")
    print(f"Images missing a mask:            {len(missing_masks)}")

    if missing_masks:
        print("\nMissing masks:")
        for line in missing_masks:
            print(f"  - {line}")

    print(f"\nFAUNA CHECK: {'PASSED' if passed else 'FAILED'}")
    print()

    return passed, missing_masks, total_images


# ---------------------------------------------------------------------------
# PART 2: MANGROVE CHECK (COCO JSON)
# ---------------------------------------------------------------------------
def load_coco_file_names(json_path: Path):
    """
    Opens one COCO JSON file and returns the set of file_name values
    listed in its "images" section. Returns an empty set (and prints a
    warning) if the file is missing or can't be parsed.
    """
    if not json_path.exists():
        print(f"Warning: JSON file not found: {json_path}")
        return set()

    try:
        with open(json_path, "r") as f:
            coco_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse {json_path.name} ({e})")
        return set()

    images_list = coco_data.get("images", [])
    file_names = {img["file_name"] for img in images_list if "file_name" in img}
    return file_names


def check_mangrove_labels():
    """
    Returns (passed: bool, images_not_in_json: list[str],
              json_entries_missing_on_disk: list[str], total_images: int)
    """
    print("=" * 70)
    print("CHECKING MANGROVE IMAGES (data/mangrove_frames/) against COCO JSON")
    print("=" * 70)

    if not MANGROVE_IMAGES_DIR.exists():
        print(f"ERROR: folder not found: {MANGROVE_IMAGES_DIR}")
        return False, [], [], 0

    # Collect all image files actually sitting in mangrove_frames/
    mangrove_images = sorted(
        f for f in MANGROVE_IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    images_on_disk = {f.name for f in mangrove_images}

    # Combine the file_name entries from ALL three JSON files into one set.
    # "At least one file" means: if it's in train OR valid OR test, that's fine.
    all_json_file_names = set()
    for json_path in MANGROVE_JSON_FILES:
        names = load_coco_file_names(json_path)
        print(f"  {json_path.name}: {len(names)} image entries")
        all_json_file_names |= names

    # Direction A: every image on disk must appear in at least one JSON
    images_not_in_json = sorted(images_on_disk - all_json_file_names)

    # Direction B: every file_name mentioned in the JSONs must exist on disk
    json_entries_missing_on_disk = sorted(all_json_file_names - images_on_disk)

    total_images = len(mangrove_images)
    passed = len(images_not_in_json) == 0 and len(json_entries_missing_on_disk) == 0

    # --- Print results ---
    print(f"\nImages found in data/mangrove_frames/:   {total_images}")
    print(f"Unique images referenced across JSONs:   {len(all_json_file_names)}")
    print(f"Images on disk but NOT in any JSON:      {len(images_not_in_json)}")
    print(f"JSON entries with NO matching file:      {len(json_entries_missing_on_disk)}")

    if images_not_in_json:
        print("\nImages on disk but missing from all JSON files:")
        for name in images_not_in_json:
            print(f"  - {name}")

    if json_entries_missing_on_disk:
        print("\nJSON entries pointing to files that don't exist on disk:")
        for name in json_entries_missing_on_disk:
            print(f"  - {name}")

    print(f"\nMANGROVE CHECK: {'PASSED' if passed else 'FAILED'}")
    print()

    return passed, images_not_in_json, json_entries_missing_on_disk, total_images


# ---------------------------------------------------------------------------
# PART 3: PACKAGED DATASET CHECK (dataset/ train/val/test)
# ---------------------------------------------------------------------------
def check_packaged_dataset():
    """
    Checks the final packaged dataset/ directory structure, ensuring splits
    are non-empty, .gitkeep files are removed, and all images have corresponding
    annotations (fauna PNG masks or mangrove COCO JSON entries).
    """
    dataset_dir = PROJECT_ROOT / "dataset"
    if not dataset_dir.exists():
        return True, "Skipped (dataset/ folder does not exist yet)"

    print("=" * 70)
    print("CHECKING PACKAGED DATASET (dataset/ train/val/test splits)")
    print("=" * 70)

    splits = ["train", "val", "test"]
    issues = []
    total_packaged_images = 0

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            issues.append(f"Missing split directory: dataset/{split}")
            continue

        gitkeep = split_dir / ".gitkeep"
        if gitkeep.exists():
            issues.append(f"Leftover .gitkeep file found in dataset/{split}/.gitkeep")

        img_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"

        if not img_dir.exists() or not any(img_dir.iterdir()):
            issues.append(f"Empty or missing images directory in dataset/{split}/images")
            continue
        if not ann_dir.exists():
            issues.append(f"Missing annotations directory in dataset/{split}/annotations")
            continue

        images = sorted(f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        total_packaged_images += len(images)

        # Collect fauna masks and COCO JSON entries
        fauna_masks = {f.name for f in ann_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"}
        
        coco_file = ann_dir / "mangrove_annotations.coco.json"
        coco_images = set()
        if coco_file.exists():
            coco_images = load_coco_file_names(coco_file)

        for img_path in images:
            fname = img_path.name
            if "img_" in fname:  # Mangrove image
                if fname not in coco_images:
                    issues.append(f"Mangrove image {fname} in dataset/{split} missing from COCO JSON")
            else:  # Fauna image (suim, deepfish, f4k)
                base_stem = fname.split("_turb")[0]
                expected_mask = f"{base_stem}_mask.png"
                if expected_mask not in fauna_masks:
                    issues.append(f"Fauna image {fname} in dataset/{split} missing mask {expected_mask}")

    passed = len(issues) == 0
    print(f"Total packaged synthetic images checked: {total_packaged_images}")
    print(f"Issues detected in dataset/ package:     {len(issues)}")
    if issues:
        print("\nPackaged Dataset Issues:")
        for line in issues[:10]:
            print(f"  - {line}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues.")

    print(f"\nPACKAGED DATASET CHECK: {'PASSED' if passed else 'FAILED'}\n")
    return passed, total_packaged_images


# ---------------------------------------------------------------------------
# MAIN: run both checks and print one final verdict
# ---------------------------------------------------------------------------
def main():
    print("\nVERIFY LABELS — Turbid Water Dataset Project")
    print(f"Project root: {PROJECT_ROOT}\n")

    fauna_passed, missing_masks, fauna_total = check_fauna_labels()
    mangrove_passed, missing_from_json, missing_on_disk, mangrove_total = check_mangrove_labels()
    packaged_passed, packaged_total = check_packaged_dataset()

    overall_passed = fauna_passed and mangrove_passed and (packaged_passed is True)

    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print(f"Fauna (raw):    {'PASSED' if fauna_passed else 'FAILED'}  "
          f"({fauna_total - len(missing_masks)}/{fauna_total} images have masks)")
    print(f"Mangrove (raw): {'PASSED' if mangrove_passed else 'FAILED'}  "
          f"({mangrove_total - len(missing_from_json)}/{mangrove_total} images labeled, "
          f"{len(missing_on_disk)} JSON entries missing on disk)")
    if isinstance(packaged_passed, bool):
        print(f"Packaged:       {'PASSED' if packaged_passed else 'FAILED'}  "
              f"({packaged_total} packaged synthetic images verified)")
    print()
    print(f"OVERALL: {'PASSED ✅' if overall_passed else 'FAILED ❌'}")
    print("=" * 70)

    # Exit code 0 = success, 1 = failure. Useful if you ever wire this
    # into an automated pipeline or CI check.
    return 0 if overall_passed else 1


if __name__ == "__main__":
    exit(main())
