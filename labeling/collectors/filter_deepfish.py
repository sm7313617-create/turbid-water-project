"""
DeepFish Dataset Filtering and Ingestion Collector:
--------------------------------------------------
Extract fish-containing image/mask pairs from the DeepFish Segmentation dataset
(located under images/valid/ and masks/valid/) and copy them into this project's
data directories (data/raw and data/annotations) with sequential renaming.
"""

import os
import sys
import argparse
import shutil
import csv
import numpy as np
from PIL import Image


def filter_deepfish(
    source_dir: str,
    target_count: int = 200,
    min_fish_pixels: int = 0,
    dry_run: bool = False,
    output_raw_dir: str = "data/raw",
    output_annotations_dir: str = "data/annotations/fauna/deepfish_masks",
    log_dir: str = "labeling/collectors/logs",
):
    """
    Filter the DeepFish Segmentation dataset for valid fish-containing images
    and copy them sequentially renamed to target output folders.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    images_dir = os.path.join(source_dir, "images", "valid")
    masks_dir = os.path.join(source_dir, "masks", "valid")

    if not os.path.isdir(images_dir):
        print(f"Error: Images directory '{images_dir}' not found.")
        sys.exit(1)

    if not os.path.isdir(masks_dir):
        print(f"Error: Masks directory '{masks_dir}' not found.")
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)
    skipped_log_path = os.path.join(log_dir, "deepfish_skipped_log.csv")
    selection_log_path = os.path.join(log_dir, "deepfish_selection_log.csv")

    skipped_rows = []
    selection_rows = []

    if not dry_run:
        os.makedirs(output_raw_dir, exist_ok=True)
        os.makedirs(output_annotations_dir, exist_ok=True)

    # Map mask filename stems (case-insensitive) to actual file names
    mask_files = os.listdir(masks_dir)
    mask_map = {}
    for m_name in mask_files:
        stem = os.path.splitext(m_name)[0]
        mask_map[stem.lower()] = m_name

    # Walk through image files sorted for deterministic processing
    image_files = sorted(os.listdir(images_dir))
    copied_count = 0

    print(f"Source directory: {source_dir}")
    print(f"Target count: {target_count}, Min fish pixels: {min_fish_pixels}, Dry run: {dry_run}")
    print("-" * 70)

    for img_name in image_files:
        if copied_count >= target_count:
            break

        ext = os.path.splitext(img_name)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        stem = os.path.splitext(img_name)[0]
        matching_mask_name = mask_map.get(stem.lower())

        img_path = os.path.join(images_dir, img_name)

        if not matching_mask_name:
            reason = "no matching mask"
            print(f"Skipping {img_name}: {reason}")
            skipped_rows.append({
                "original_filename": img_name,
                "reason": reason,
            })
            continue

        mask_path = os.path.join(masks_dir, matching_mask_name)

        fish_pixel_count = None
        if min_fish_pixels > 0:
            try:
                with Image.open(mask_path) as mask_img:
                    mask_arr = np.array(mask_img)

                fish_pixel_count = int(np.count_nonzero(mask_arr > 0))

            except Exception as e:
                reason = f"corrupt or unreadable mask ({e})"
                print(f"Skipping {img_name}: {reason}")
                skipped_rows.append({
                    "original_filename": img_name,
                    "reason": reason,
                })
                continue

            if fish_pixel_count < min_fish_pixels:
                reason = f"below threshold ({fish_pixel_count} < {min_fish_pixels} fish pixels)"
                skipped_rows.append({
                    "original_filename": img_name,
                    "reason": reason,
                })
                continue

        # Qualifying image pair found
        copied_count += 1
        seq_id = f"deepfish_{copied_count:05d}"
        new_img_name = f"{seq_id}.jpg"
        new_mask_name = f"{seq_id}_mask.png"

        log_msg = f"[{copied_count}/{target_count}] Found {img_name} -> {seq_id}"
        if fish_pixel_count is not None:
            log_msg += f" (Fish pixels: {fish_pixel_count})"
        print(log_msg)

        if not dry_run:
            dest_img_path = os.path.join(output_raw_dir, new_img_name)
            dest_mask_path = os.path.join(output_annotations_dir, new_mask_name)

            shutil.copy2(img_path, dest_img_path)
            shutil.copy2(mask_path, dest_mask_path)

        selection_rows.append({
            "original_filename": img_name,
            "new_filename": new_img_name,
        })

    # Write skipped log CSV (written in both dry-run and normal mode)
    with open(skipped_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["original_filename", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"\nSkipped log written to: {skipped_log_path} ({len(skipped_rows)} entries)")

    if not dry_run:
        with open(selection_log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["original_filename", "new_filename"]
            )
            writer.writeheader()
            writer.writerows(selection_rows)
        print(f"Selection log written to: {selection_log_path} ({len(selection_rows)} entries)")

    print("-" * 70)
    if copied_count < target_count:
        print(f"WARNING: Only found {copied_count} qualifying images matching criteria (target requested was {target_count}).")
    else:
        print(f"SUCCESS: Successfully processed {copied_count} qualifying image pairs.")

    if dry_run:
        print(f"[DRY-RUN COMPLETE] Total qualifying images found: {copied_count}. No files were copied.")


def main():
    parser = argparse.ArgumentParser(
        description="Filter DeepFish Segmentation dataset for fish-containing images and copy with sequential renaming."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to DeepFish Segmentation dataset root directory (containing images/valid and masks/valid folders)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=200,
        help="Target number of qualifying images to extract (default: 200)",
    )
    parser.add_argument(
        "--min-fish-pixels",
        type=int,
        default=0,
        help="Minimum required white fish pixels in mask (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scan and write deepfish_skipped_log.csv without copying files to data directories",
    )

    args = parser.parse_args()

    filter_deepfish(
        source_dir=args.source,
        target_count=args.target_count,
        min_fish_pixels=args.min_fish_pixels,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
