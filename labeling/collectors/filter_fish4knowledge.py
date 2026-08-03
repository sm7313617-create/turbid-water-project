"""
Fish4Knowledge Dataset Filtering and Ingestion Collector:
--------------------------------------------------------
Extract fish image/mask pairs from multiple species cluster folders in the
Fish4Knowledge Recognition dataset (located under fish_XX/ and mask_XX/)
and copy them into this project's data directories (data/raw and data/annotations)
with sequential renaming (f4k_XXXXX.jpg and f4k_XXXXX_mask.png).
"""

import os
import sys
import argparse
import shutil
import csv
import numpy as np
from PIL import Image


def filter_fish4knowledge(
    source_dir: str,
    target_count: int = 150,
    per_cluster_count: int = None,
    min_fish_pixels: int = 0,
    dry_run: bool = False,
    output_raw_dir: str = "data/raw",
    output_annotations_dir: str = "data/annotations/fauna/f4k_masks",
    log_dir: str = "labeling/collectors/logs",
):
    """
    Filter the Fish4Knowledge Recognition dataset across cluster subfolders
    for valid fish-containing image/mask pairs, converting raw images to JPG
    and sequentially renaming into output folders.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Auto-detect all cluster subfolders matching 'fish_*' that have a corresponding 'mask_*' folder
    all_entries = os.listdir(source_dir)
    cluster_pairs = []

    for entry in sorted(all_entries):
        if entry.startswith("fish_"):
            cluster_id = entry[5:]
            fish_dir = os.path.join(source_dir, entry)
            mask_dir = os.path.join(source_dir, f"mask_{cluster_id}")
            if os.path.isdir(fish_dir):
                if os.path.isdir(mask_dir):
                    cluster_pairs.append((cluster_id, fish_dir, mask_dir))
                else:
                    print(f"Warning: Matching mask directory '{mask_dir}' not found for '{entry}'. Skipping cluster.")

    if not cluster_pairs:
        print(f"Error: No valid 'fish_XX' and 'mask_XX' subfolder pairs found in '{source_dir}'.")
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)
    skipped_log_path = os.path.join(log_dir, "f4k_skipped_log.csv")
    selection_log_path = os.path.join(log_dir, "f4k_selection_log.csv")

    skipped_rows = []
    selection_rows = []

    if not dry_run:
        os.makedirs(output_raw_dir, exist_ok=True)
        os.makedirs(output_annotations_dir, exist_ok=True)

    copied_count = 0
    cluster_summary = {}

    print(f"Source directory: {source_dir}")
    print(f"Found {len(cluster_pairs)} cluster pair(s): {[c[0] for c in cluster_pairs]}")

    if per_cluster_count is not None:
        expected_total = per_cluster_count * len(cluster_pairs)
        print(f"Per-cluster target: {per_cluster_count} (Total target: {expected_total}), Min fish pixels: {min_fish_pixels}, Dry run: {dry_run}")
    else:
        expected_total = target_count
        print(f"Target count: {target_count}, Min fish pixels: {min_fish_pixels}, Dry run: {dry_run}")

    print("-" * 70)

    for cluster_id, fish_dir, mask_dir in cluster_pairs:
        if per_cluster_count is None and copied_count >= target_count:
            break

        cluster_copied = 0

        # Build lookup map for mask files by stripping 'mask_' prefix from stem
        mask_files = os.listdir(mask_dir)
        mask_map = {}
        for m_name in mask_files:
            m_stem = os.path.splitext(m_name)[0]
            if m_stem.lower().startswith("mask_"):
                shared_id = m_stem[5:].lower()
            else:
                shared_id = m_stem.lower()
            mask_map[shared_id] = m_name

        image_files = sorted(os.listdir(fish_dir))

        for img_name in image_files:
            if per_cluster_count is not None:
                if cluster_copied >= per_cluster_count:
                    break
            else:
                if copied_count >= target_count:
                    break

            ext = os.path.splitext(img_name)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            img_stem = os.path.splitext(img_name)[0]
            if img_stem.lower().startswith("fish_"):
                shared_id = img_stem[5:].lower()
            else:
                shared_id = img_stem.lower()

            matching_mask_name = mask_map.get(shared_id)
            img_path = os.path.join(fish_dir, img_name)

            if not matching_mask_name:
                reason = "no matching mask"
                print(f"Skipping cluster {cluster_id}/{img_name}: {reason}")
                skipped_rows.append({
                    "cluster_id": cluster_id,
                    "original_filename": img_name,
                    "reason": reason,
                })
                continue

            mask_path = os.path.join(mask_dir, matching_mask_name)

            fish_pixel_count = None
            if min_fish_pixels > 0:
                try:
                    with Image.open(mask_path) as mask_img:
                        mask_arr = np.array(mask_img)

                    fish_pixel_count = int(np.count_nonzero(mask_arr > 0))

                except Exception as e:
                    reason = f"corrupt or unreadable mask ({e})"
                    print(f"Skipping cluster {cluster_id}/{img_name}: {reason}")
                    skipped_rows.append({
                        "cluster_id": cluster_id,
                        "original_filename": img_name,
                        "reason": reason,
                    })
                    continue

                if fish_pixel_count < min_fish_pixels:
                    reason = f"below threshold ({fish_pixel_count} < {min_fish_pixels} fish pixels)"
                    skipped_rows.append({
                        "cluster_id": cluster_id,
                        "original_filename": img_name,
                        "reason": reason,
                    })
                    continue

            # Qualifying image pair found
            copied_count += 1
            cluster_copied += 1
            seq_id = f"f4k_{copied_count:05d}"
            new_img_name = f"{seq_id}.jpg"
            new_mask_name = f"{seq_id}_mask.png"

            log_msg = f"[{copied_count}/{expected_total}] Found cluster {cluster_id}/{img_name} -> {seq_id}"
            if fish_pixel_count is not None:
                log_msg += f" (Fish pixels: {fish_pixel_count})"
            print(log_msg)

            if not dry_run:
                dest_img_path = os.path.join(output_raw_dir, new_img_name)
                dest_mask_path = os.path.join(output_annotations_dir, new_mask_name)

                # Convert PNG image to JPG
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.save(dest_img_path, format="JPEG", quality=95)

                # Copy binary mask PNG directly
                shutil.copy2(mask_path, dest_mask_path)

            selection_rows.append({
                "cluster_id": cluster_id,
                "original_filename": img_name,
                "new_filename": new_img_name,
            })

        if per_cluster_count is not None:
            if cluster_copied < per_cluster_count:
                print(f"WARNING: Cluster {cluster_id} provided only {cluster_copied} qualifying images (shortfall: {per_cluster_count - cluster_copied}).")
            cluster_summary[cluster_id] = f"{cluster_copied}/{per_cluster_count} copied"
        else:
            cluster_summary[cluster_id] = f"{cluster_copied} copied"

    # Write skipped log CSV (written in both dry-run and normal mode)
    with open(skipped_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "original_filename", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"\nSkipped log written to: {skipped_log_path} ({len(skipped_rows)} entries)")

    if not dry_run:
        with open(selection_log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["cluster_id", "original_filename", "new_filename"]
            )
            writer.writeheader()
            writer.writerows(selection_rows)
        print(f"Selection log written to: {selection_log_path} ({len(selection_rows)} entries)")

    print("-" * 70)
    summary_parts = [f"cluster {cid}: {info}" for cid, info in cluster_summary.items()]
    print("Per-cluster summary: " + ", ".join(summary_parts))

    if copied_count < expected_total:
        print(f"WARNING: Only found {copied_count} qualifying images matching criteria (total requested was {expected_total}).")
    else:
        print(f"SUCCESS: Successfully processed {copied_count} qualifying image pairs.")

    if dry_run:
        print(f"[DRY-RUN COMPLETE] Total qualifying images found: {copied_count}. No files were copied.")


def main():
    parser = argparse.ArgumentParser(
        description="Filter Fish4Knowledge dataset across cluster subfolders for fish images and copy with sequential renaming."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to Fish4Knowledge dataset root directory (containing fish_XX and mask_XX cluster folders)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=150,
        help="Target number of qualifying images to extract across all clusters (default: 150)",
    )
    parser.add_argument(
        "--per-cluster-count",
        type=int,
        default=None,
        help="Number of qualifying images to extract from EACH cluster independently (overrides --target-count)",
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
        help="Run scan and write f4k_skipped_log.csv without copying files to data directories",
    )

    args = parser.parse_args()

    filter_fish4knowledge(
        source_dir=args.source,
        target_count=args.target_count,
        per_cluster_count=args.per_cluster_count,
        min_fish_pixels=args.min_fish_pixels,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
