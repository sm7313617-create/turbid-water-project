"""
audit_fauna_contamination.py
------------------------------
Perceptual similarity audit tool for detecting mangrove contamination (Type A)
and internal near-duplicate clusters (Type B) in data/raw/ using imagehash (phash).

DISTINCTION OF CHECK TYPES:
----------------------------
- Type A (Mangrove match): Image in data/raw/ matches real footage in data/mangrove_frames/
  or known mangrove reference images (Hamming distance <= 8). Indicates actual label contamination.
- Type B (Internal duplicate only): Image closely matches another image within its own source
  (e.g., deepfish_00014 vs deepfish_00015) but has NO match to mangrove footage. Indicates
  sequential/video frame redundancy inherent to the source dataset.
- Clean: No perceptual similarity flags.

OUTPUT:
-------
1. Prints a 4-column breakdown table (Source | Total | Type A | Type B | Clean).
2. Generates a visual contact sheet in audit_sanity_check/index.html for manual inspection.
"""

import os
import sys
import glob
import io
import random
import shutil
import json
from pathlib import Path
from PIL import Image

import imagehash

# Ensure UTF-8 output encoding for Windows terminals
if isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANGROVE_DIR = PROJECT_ROOT / "data" / "mangrove_frames"
OUTPUT_SANITY_DIR = PROJECT_ROOT / "audit_sanity_check"

KNOWN_BAD_REFERENCES = [
    "deepfish_00001",
    "deepfish_00002",
    "deepfish_00003",
    "deepfish_00007"
]

SIMILARITY_THRESHOLD = 8


def compute_phash(image_path: Path):
    """
    Opens an image file and computes its 64-bit perceptual hash (pHash).
    """
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"Error reading image {image_path.name}: {e}")
        return None


def run_audit():
    """
    Runs the pHash similarity audit across DeepFish, SUIM, and F4K.
    Classifies every image as Type A (Mangrove match), Type B (Internal dup only), or Clean.
    """
    print("=" * 70)
    print("RUNNING PERCEPTUAL AUDIT (TYPE A vs TYPE B CLASSIFICATION)")
    print("=" * 70)

    # 1. Precompute mangrove reference hashes
    mangrove_files = sorted(list(MANGROVE_DIR.glob("*.*")))
    mangrove_hashes = {}
    for mp in mangrove_files:
        mh = compute_phash(mp)
        if mh is not None:
            mangrove_hashes[mp.name] = (mp, mh)

    # Include known bad reference DeepFish files as mangrove targets
    ref_bad_hashes = {}
    for stem in KNOWN_BAD_REFERENCES:
        ref_path = RAW_DIR / f"{stem}.jpg"
        if ref_path.exists():
            rh = compute_phash(ref_path)
            if rh is not None:
                ref_bad_hashes[f"{stem}.jpg"] = (ref_path, rh)

    # All mangrove reference targets (real mangrove frames + known bad references)
    all_mangrove_targets = {**mangrove_hashes, **ref_bad_hashes}

    sources = ["deepfish", "suim", "f4k"]
    audit_results = {}

    for src in sources:
        src_files = sorted(list(RAW_DIR.glob(f"{src}_*.*")))
        src_hashes = {}
        for p in src_files:
            h = compute_phash(p)
            if h is not None:
                src_hashes[p.stem] = (p, h)

        type_a_matches = {}  # stem -> (matched_mangrove_name, matched_mangrove_path, distance)
        type_b_matches = {}  # stem -> (matched_internal_stem, matched_internal_path, distance)

        # Check Type A: Mangrove match
        for stem, (path, h) in src_hashes.items():
            best_mang_name = None
            best_mang_path = None
            min_mang_dist = 999
            for mang_name, (mang_path, mang_h) in all_mangrove_targets.items():
                dist = h - mang_h
                if dist < min_mang_dist:
                    min_mang_dist = dist
                    best_mang_name = mang_name
                    best_mang_path = mang_path
            
            if min_mang_dist <= SIMILARITY_THRESHOLD:
                type_a_matches[stem] = (best_mang_name, best_mang_path, min_mang_dist)

        # Check Type B: Internal duplicate only
        stems_list = sorted(list(src_hashes.keys()))
        for i in range(len(stems_list)):
            for j in range(i + 1, len(stems_list)):
                stem1, stem2 = stems_list[i], stems_list[j]
                p1, h1 = src_hashes[stem1]
                p2, h2 = src_hashes[stem2]
                dist = h1 - h2

                if dist <= SIMILARITY_THRESHOLD:
                    if stem1 not in type_a_matches and stem1 not in type_b_matches:
                        type_b_matches[stem1] = (stem2, p2, dist)
                    if stem2 not in type_a_matches and stem2 not in type_b_matches:
                        type_b_matches[stem2] = (stem1, p1, dist)

        total_count = len(src_files)
        type_a_stems = set(type_a_matches.keys())
        type_b_stems = set(type_b_matches.keys())
        clean_stems = set(src_hashes.keys()) - type_a_stems - type_b_stems

        audit_results[src] = {
            "total": total_count,
            "type_a_matches": type_a_matches,
            "type_b_matches": type_b_matches,
            "clean_stems": clean_stems,
            "src_hashes": src_hashes,
        }

    return audit_results


def print_summary_table(audit_results):
    """
    Prints a clear 5-column breakdown table as requested.
    """
    print("\n" + "=" * 70)
    print("FAUNA CONTAMINATION BREAKDOWN SUMMARY")
    print("=" * 70 + "\n")

    header = f"| {'Source':<10} | {'Total':<6} | {'Type A (mangrove match)':<23} | {'Type B (internal dup only)':<26} | {'Clean':<6} |"
    divider = f"|{'-'*12}|{'-'*8}|{'-'*25}|{'-'*28}|{'-'*8}|"

    print(header)
    print(divider)

    total_all = 0
    total_a = 0
    total_b = 0
    total_clean = 0

    for src in ["deepfish", "suim", "f4k"]:
        res = audit_results[src]
        t = res["total"]
        a = len(res["type_a_matches"])
        b = len(res["type_b_matches"])
        c = len(res["clean_stems"])

        total_all += t
        total_a += a
        total_b += b
        total_clean += c

        print(f"| {src:<10} | {t:<6d} | {a:<23d} | {b:<26d} | {c:<6d} |")

    print(divider)
    print(f"| {'TOTAL':<10} | {total_all:<6d} | {total_a:<23d} | {total_b:<26d} | {total_clean:<6d} |")
    print("=" * 70 + "\n")


def generate_visual_sanity_check_sheet(audit_results, seed=42):
    """
    Generates a visual contact sheet in audit_sanity_check/index.html with
    sampled image pairs for Type A and Type B flags.
    """
    print("Generating visual sanity check HTML report in audit_sanity_check/...")

    if OUTPUT_SANITY_DIR.exists():
        shutil.rmtree(OUTPUT_SANITY_DIR)

    img_out_dir = OUTPUT_SANITY_DIR / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # 1. Sample 10 Type A DeepFish
    df_res = audit_results["deepfish"]
    type_a_df_stems = sorted(list(df_res["type_a_matches"].keys()))
    sample_df_type_a = rng.sample(type_a_df_stems, min(10, len(type_a_df_stems)))

    # 2. Sample 10 Type A F4K
    f4k_res = audit_results["f4k"]
    type_a_f4k_stems = sorted(list(f4k_res["type_a_matches"].keys()))
    sample_f4k_type_a = rng.sample(type_a_f4k_stems, min(10, len(type_a_f4k_stems)))

    # 3. Sample 10 Type B DeepFish
    type_b_df_stems = sorted(list(df_res["type_b_matches"].keys()))
    sample_df_type_b = rng.sample(type_b_df_stems, min(10, len(type_b_df_stems)))

    # 4. Sample 10 Type B F4K
    type_b_f4k_stems = sorted(list(f4k_res["type_b_matches"].keys()))
    sample_f4k_type_b = rng.sample(type_b_f4k_stems, min(10, len(type_b_f4k_stems)))

    def copy_thumbnail(src_path, filename):
        dst_path = img_out_dir / filename
        shutil.copy2(src_path, dst_path)
        return f"images/{filename}"

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fauna Contamination & Near-Duplicate Visual Sanity Check</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; line-height: 1.5; }
        h1 { color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }
        h2 { color: #f43f5e; margin-top: 40px; border-bottom: 1px solid #334155; padding-bottom: 5px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 20px; margin-top: 15px; }
        .card { background: #1e293b; border-radius: 8px; border: 1px solid #334155; padding: 15px; }
        .pair-title { font-weight: bold; font-size: 14px; margin-bottom: 10px; color: #fbbf24; }
        .images { display: flex; gap: 10px; justify-content: space-around; }
        .img-box { text-align: center; flex: 1; }
        .img-box img { max-width: 100%; height: 160px; object-fit: cover; border-radius: 4px; border: 1px solid #475569; }
        .img-label { font-size: 12px; color: #94a3b8; margin-top: 5px; word-break: break-all; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; background: #0284c7; color: white; margin-bottom: 8px; }
        .dist { color: #34d399; font-weight: bold; }
        .note { background: #334155; padding: 10px 15px; border-radius: 6px; margin-bottom: 20px; font-size: 14px; }
    </style>
</head>
<body>
    <h1>Visual Sanity Check: Fauna Contamination & Near-Duplicates</h1>
    <div class="note">
        This sheet presents side-by-side image comparisons to visually verify perceptual hashing (pHash) flags before taking any dataset modification actions.
    </div>
"""

    def render_section(section_title, sample_stems, matches_dict, src_hashes_dict, badge_label, is_type_a):
        nonlocal html_content
        html_content += f"<h2>{section_title} ({len(sample_stems)} sampled pairs)</h2>\n"
        if not sample_stems:
            html_content += "<p style='color:#94a3b8;'>No matches found in this category.</p>\n"
            return

        html_content += '<div class="grid">\n'
        for idx, stem in enumerate(sample_stems):
            src_path, _ = src_hashes_dict[stem]
            matched_name, matched_path, dist = matches_dict[stem]

            src_img_rel = copy_thumbnail(src_path, f"{stem}_src.jpg")
            match_img_rel = copy_thumbnail(matched_path, f"{stem}_match.jpg")

            html_content += f"""
            <div class="card">
                <div class="badge">{badge_label} #{idx+1}</div>
                <div class="pair-title">Stem: {stem} &bull; Distance: <span class="dist">{dist} bits</span></div>
                <div class="images">
                    <div class="img-box">
                        <img src="{src_img_rel}" alt="{stem}">
                        <div class="img-label">Target Fauna: {stem}</div>
                    </div>
                    <div class="img-box">
                        <img src="{match_img_rel}" alt="{matched_name}">
                        <div class="img-label">Matched: {matched_name}</div>
                    </div>
                </div>
            </div>
            """
        html_content += '</div>\n'

    # Section 1: Type A DeepFish
    render_section(
        "Section 1: Type A — DeepFish vs Mangrove Footage Matches",
        sample_df_type_a,
        df_res["type_a_matches"],
        df_res["src_hashes"],
        "Type A (Mangrove Contamination)",
        is_type_a=True,
    )

    # Section 2: Type A F4K
    render_section(
        "Section 2: Type A — F4K vs Mangrove Footage Matches",
        sample_f4k_type_a,
        f4k_res["type_a_matches"],
        f4k_res["src_hashes"],
        "Type A (Mangrove Contamination)",
        is_type_a=True,
    )

    # Section 3: Type B DeepFish
    render_section(
        "Section 3: Type B — DeepFish Internal Near-Duplicates",
        sample_df_type_b,
        df_res["type_b_matches"],
        df_res["src_hashes"],
        "Type B (Internal Near-Duplicate)",
        is_type_a=False,
    )

    # Section 4: Type B F4K
    render_section(
        "Section 4: Type B — F4K Internal Near-Duplicates",
        sample_f4k_type_b,
        f4k_res["type_b_matches"],
        f4k_res["src_hashes"],
        "Type B (Internal Near-Duplicate)",
        is_type_a=False,
    )

    html_content += """
</body>
</html>
"""

    index_html_path = OUTPUT_SANITY_DIR / "index.html"
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Visual sanity check report generated at: {index_html_path.resolve()}")


def main():
    audit_results = run_audit()
    print_summary_table(audit_results)
    generate_visual_sanity_check_sheet(audit_results, seed=42)


if __name__ == "__main__":
    main()
