"""
generate.py - Batch Underwater Synthetic Dataset Generator

This script batch-processes aquatic image datasets (e.g. clean fauna frames in data/raw
and mangrove frames in data/mangrove_frames) across multiple turbidity levels using the
physically-grounded degradation model.

Features:
---------
1. Multi-level batch generation (default levels: 0.2, 0.4, 0.6, 0.8).
2. Traceable naming convention: {original_stem}_turb{level}.png so ground truth
   annotations (mask PNGs or COCO JSON entries) remain directly linked.
3. Windows case-insensitive deduplication using os.listdir + ext matching.
4. Progress tracking via tqdm progress bars.
5. Generates a dataset manifest CSV (synthetic_manifest.csv) in data/synthetic/.
6. Configurable CLI flags (argparse), including --max-images for small spot-check runs.
"""

import os
import sys
import argparse
import csv

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", unit=""):
        print(f"Running {desc}...")
        return iterable

# Handle imports when run as script or module
try:
    from generator.utils import load_image, save_image
    from generator.degradation import degrade_image
except ImportError:
    from utils import load_image, save_image
    from degradation import degrade_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generate physically-degraded underwater images at multiple turbidity levels."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["data/raw", "data/mangrove_frames"],
        help="Input directories containing clean source images (default: data/raw data/mangrove_frames).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Target output directory for synthetic images (default: data/synthetic).",
    )
    parser.add_argument(
        "--turbidity-levels",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8],
        help="Turbidity levels to simulate in range [0.0, 1.0] (default: 0.2 0.4 0.6 0.8).",
    )
    parser.add_argument(
        "--depth-mode",
        type=str,
        default="gradient",
        choices=["gradient", "radial"],
        help="Synthetic depth profile mode ('gradient' or 'radial', default: gradient).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of source images to process per directory (useful for small test runs).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing synthetic output files if present.",
    )
    return parser.parse_args()


def batch_generate(
    input_dirs: list,
    output_dir: str = "data/synthetic",
    turbidity_levels: list = [0.2, 0.4, 0.6, 0.8],
    depth_mode: str = "gradient",
    max_images: int | None = None,
    overwrite: bool = False,
):
    """
    Batch generate synthetic turbid images across multiple input directories and turbidity levels.

    Args:
        input_dirs (list): List of input folder paths containing source images.
        output_dir (str): Target directory for synthetic output images.
        turbidity_levels (list): List of float turbidity levels in range [0.0, 1.0].
        depth_mode (str): Synthetic depth profile ('gradient' or 'radial').
        max_images (int, optional): Max images to process per input directory.
        overwrite (bool): If True, re-generates and overwrites existing synthetic images.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "synthetic_manifest.csv")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_image_paths = []

    print("\n==================================================")
    print("--- INGESTING SOURCE IMAGE DIRECTORIES ---")
    print("==================================================")

    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory '{input_dir}' does not exist. Skipping.")
            continue

        dir_files = []
        for filename in os.listdir(input_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.normpath(os.path.join(input_dir, filename))
                dir_files.append(full_path)

        dir_files = sorted(dir_files)
        count_in_dir = len(dir_files)

        if max_images is not None and max_images > 0:
            dir_files_to_process = dir_files[:max_images]
            print(f"  [+] Directory '{input_dir}': {count_in_dir} unique images found (limiting to first {len(dir_files_to_process)} for test run).")
        else:
            dir_files_to_process = dir_files
            print(f"  [+] Directory '{input_dir}': {count_in_dir} unique images found.")

        all_image_paths.extend(dir_files_to_process)

    total_images = len(all_image_paths)

    if total_images == 0:
        print("No source images found in specified input directories. Aborting.")
        return

    total_synthetic_count = total_images * len(turbidity_levels)
    print(f"\n==================================================")
    print(f"--- BATCH SYNTHETIC GENERATION PLAN ---")
    print(f"==================================================")
    print(f"Source Directories : {input_dirs}")
    print(f"Total Source Images: {total_images}")
    print(f"Turbidity Levels   : {turbidity_levels}")
    print(f"Depth Profile Mode : '{depth_mode}'")
    print(f"Target Output Dir  : '{output_dir}'")
    print(f"Total Outputs      : {total_synthetic_count} synthetic images")
    print(f"==================================================\n")

    manifest_rows = []

    for img_path in tqdm(all_image_paths, desc="Processing Source Images", unit="img"):
        filename = os.path.basename(img_path)
        stem, ext = os.path.splitext(filename)
        parent_folder = os.path.basename(os.path.dirname(img_path))

        try:
            raw_img = load_image(img_path)
        except Exception as e:
            print(f"\nError loading image '{img_path}': {e}. Skipping.")
            continue

        for turb in turbidity_levels:
            out_filename = f"{stem}_turb{turb:.1f}.png"
            out_path = os.path.join(output_dir, out_filename)

            manifest_rows.append(
                {
                    "synthetic_filename": out_filename,
                    "original_filename": filename,
                    "original_stem": stem,
                    "source_dataset": parent_folder,
                    "turbidity_level": turb,
                    "depth_mode": depth_mode,
                }
            )

            if os.path.exists(out_path) and not overwrite:
                continue

            degraded_img = degrade_image(raw_img, turbidity_level=turb, depth_mode=depth_mode)
            save_image(degraded_img, out_path)

    fieldnames = [
        "synthetic_filename",
        "original_filename",
        "original_stem",
        "source_dataset",
        "turbidity_level",
        "depth_mode",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[+] Batch generation complete!")
    print(f"[+] Total synthetic images generated: {total_synthetic_count}")
    print(f"[+] Dataset manifest saved to: '{manifest_path}'\n")


if __name__ == "__main__":
    args = parse_args()
    batch_generate(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        turbidity_levels=args.turbidity_levels,
        depth_mode=args.depth_mode,
        max_images=args.max_images,
        overwrite=args.overwrite,
    )
