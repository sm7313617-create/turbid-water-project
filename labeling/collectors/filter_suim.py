"""
SUIM Dataset Segmentation Mask Color Coding Map (8 Classes):
-----------------------------------------------------------------------
Category Symbol Binary   RGB Color Code    Description
-----------------------------------------------------------------------
BW       000   (0, 0, 0)         Background waterbody
HD       001   (0, 0, 255)       Human divers
PF       010   (0, 255, 0)       Aquatic plants and sea-grass
WR       011   (0, 255, 255)     Wrecks and ruins
RO       100   (255, 0, 0)       Robots/instruments
RI       101   (255, 0, 255)     Reefs and invertebrates
FV       110   (255, 255, 0)     Fish and vertebrates
SR       111   (255, 255, 255)   Sea-floor and rocks
-----------------------------------------------------------------------
Confirmed FV (Fish and Vertebrates) RGB color: (255, 255, 0) [Yellow].
Source: official SUIM repository (https://github.com/IRVLab/SUIM and https://github.com/xahidbuffon/SUIM-Net)
"""

import os
import sys
import argparse
import shutil
import csv
import numpy as np
from PIL import Image

# FV class color coding: RGB (255, 255, 0)
FV_COLOR_RGB = (255, 255, 0)


def filter_suim(
    source_dir: str,
    target_count: int = 150,
    min_fv_pixels: int = 500,
    dry_run: bool = False,
    output_raw_dir: str = "data/raw",
    output_annotations_dir: str = "data/annotations/fauna/suim_masks",
    log_dir: str = "labeling/collectors/logs",
):
    """
    Filter the SUIM dataset to extract images containing fish (FV class) exceeding
    min_fv_pixels threshold and copy them sequentially renamed to target output folders.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    splits = ["train_val", "TEST"]

    os.makedirs(log_dir, exist_ok=True)
    skipped_log_path = os.path.join(log_dir, "suim_skipped_log.csv")
    selection_log_path = os.path.join(log_dir, "suim_selection_log.csv")

    skipped_rows = []
    selection_rows = []

    if not dry_run:
        os.makedirs(output_raw_dir, exist_ok=True)
        os.makedirs(output_annotations_dir, exist_ok=True)

    copied_count = 0

    print(f"Starting scan across splits: {splits}")
    print(f"Source directory: {source_dir}")
    print(f"Target count: {target_count}, Min FV pixels: {min_fv_pixels}, Dry run: {dry_run}")
    print("-" * 70)

    for split in splits:
        if copied_count >= target_count:
            break

        split_dir = os.path.join(source_dir, split)
        images_dir = os.path.join(split_dir, "images")
        masks_dir = os.path.join(split_dir, "masks")

        if not os.path.isdir(images_dir):
            print(f"Warning: Images directory '{images_dir}' not found. Skipping split '{split}'.")
            continue

        if not os.path.isdir(masks_dir):
            print(f"Warning: Masks directory '{masks_dir}' not found. Skipping split '{split}'.")
            continue

        # Map mask filename stems (case-insensitive) to actual file names
        mask_files = os.listdir(masks_dir)
        mask_map = {}
        for m_name in mask_files:
            stem = os.path.splitext(m_name)[0]
            mask_map[stem.lower()] = m_name

        # Walk through image files sorted for deterministic processing
        image_files = sorted(os.listdir(images_dir))

        for img_name in image_files:
            if copied_count >= target_count:
                break

            # Filter for common image extensions
            ext = os.path.splitext(img_name)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            stem = os.path.splitext(img_name)[0]
            matching_mask_name = mask_map.get(stem.lower())

            img_path = os.path.join(images_dir, img_name)

            if not matching_mask_name:
                reason = "no matching mask"
                print(f"Skipping {split}/{img_name}: {reason}")
                skipped_rows.append({
                    "original_split": split,
                    "original_filename": img_name,
                    "reason": reason,
                })
                continue

            mask_path = os.path.join(masks_dir, matching_mask_name)

            try:
                with Image.open(mask_path) as mask_img:
                    mask_rgb = np.array(mask_img.convert("RGB"))

                # Check FV pixels matching RGB (255, 255, 0)
                fv_mask = (
                    (mask_rgb[:, :, 0] == FV_COLOR_RGB[0]) &
                    (mask_rgb[:, :, 1] == FV_COLOR_RGB[1]) &
                    (mask_rgb[:, :, 2] == FV_COLOR_RGB[2])
                )
                fv_pixel_count = int(np.count_nonzero(fv_mask))

            except Exception as e:
                reason = f"corrupt or unreadable mask ({e})"
                print(f"Skipping {split}/{img_name}: {reason}")
                skipped_rows.append({
                    "original_split": split,
                    "original_filename": img_name,
                    "reason": reason,
                })
                continue

            if fv_pixel_count < min_fv_pixels:
                reason = f"below threshold ({fv_pixel_count} < {min_fv_pixels} FV pixels)"
                skipped_rows.append({
                    "original_split": split,
                    "original_filename": img_name,
                    "reason": reason,
                })
                continue

            # Qualifying image pair found
            copied_count += 1
            seq_id = f"suim_{copied_count:05d}"
            new_img_name = f"{seq_id}.jpg"
            new_mask_name = f"{seq_id}_mask.png"

            print(f"[{copied_count}/{target_count}] Found {split}/{img_name} -> {seq_id} (FV pixels: {fv_pixel_count})")

            if not dry_run:
                dest_img_path = os.path.join(output_raw_dir, new_img_name)
                dest_mask_path = os.path.join(output_annotations_dir, new_mask_name)

                # Copy raw image file
                shutil.copy2(img_path, dest_img_path)

                # Convert mask BMP to PNG
                with Image.open(mask_path) as m_img:
                    m_img.save(dest_mask_path, format="PNG")

                selection_rows.append({
                    "original_split": split,
                    "original_filename": img_name,
                    "new_filename": new_img_name,
                    "fv_pixel_count": fv_pixel_count,
                })

    # Write skipped log CSV (written in both dry-run and normal mode)
    with open(skipped_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["original_split", "original_filename", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"\nSkipped log written to: {skipped_log_path} ({len(skipped_rows)} entries)")

    if not dry_run:
        with open(selection_log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["original_split", "original_filename", "new_filename", "fv_pixel_count"]
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
        description="Filter SUIM dataset for images containing fish (FV class) and copy with sequential renaming."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to SUIM dataset root directory (containing train_val and TEST folders)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=150,
        help="Target number of qualifying images to extract (default: 150)",
    )
    parser.add_argument(
        "--min-fv-pixels",
        type=int,
        default=500,
        help="Minimum required FV pixel count in mask (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scan and write suim_skipped_log.csv without copying files to data directories",
    )

    args = parser.parse_args()

    filter_suim(
        source_dir=args.source,
        target_count=args.target_count,
        min_fv_pixels=args.min_fv_pixels,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
