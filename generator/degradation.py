"""
degradation.py - Physically-Grounded Underwater Image Degradation Pipeline

This module implements the core underwater optical physics simulator based on the classic
Jaffe-McGlamery Underwater Image Formation Model:

    I(x) = J(x) * t(x) + A * (1 - t(x))

Key Physical Effects Simulated:
--------------------------------
1. Selective Wavelength Absorption (Color Attenuation):
   Red light attenuates rapidly in water (high beta), green attenuates moderately,
   and blue light penetrates deepest (low beta).

2. Ambient Backscatter Haze:
   Suspended particles scatter ambient sunlight back into the camera lens. As depth and
   turbidity increase, the backscatter term A * (1 - t(x)) dominates the scene.

3. Forward Scattering (Blur & Local Contrast Reduction):
   Light bouncing off micro-particles diffuses high-frequency details, driven spatially by depth d(x).

4. Suspended Sediment (Marine Snow):
   Random bright specks simulate particle illumination near the camera lens.
"""

import os
import sys
import numpy as np

# Handle imports when run directly or as a package
try:
    from generator.utils import (
        load_image,
        save_image,
        to_float,
        to_uint8,
        clip_image,
        generate_depth_map,
        get_ambient_light_color,
        add_particle_noise,
        reduce_local_contrast,
    )
except ImportError:
    from utils import (
        load_image,
        save_image,
        to_float,
        to_uint8,
        clip_image,
        generate_depth_map,
        get_ambient_light_color,
        add_particle_noise,
        reduce_local_contrast,
    )


def get_per_channel_transmission(
    depth_map: np.ndarray,
    turbidity_level: float,
    beta_max_r: float = 3.0,
    beta_max_g: float = 1.8,
    beta_max_b: float = 1.0,
) -> np.ndarray:
    """
    Calculate 3-channel transmission maps [t_R, t_G, t_B] via Beer-Lambert's Law with
    wavelength-dependent attenuation coefficients.

    Wavelength Attenuation:
    - Red light (~700 nm) attenuates fastest -> beta_r = 3.0 * turbidity_level
    - Green light (~530 nm) attenuates moderately -> beta_g = 1.8 * turbidity_level
    - Blue light (~470 nm) penetrates deepest -> beta_b = 1.0 * turbidity_level
    """
    beta_r = turbidity_level * beta_max_r
    beta_g = turbidity_level * beta_max_g
    beta_b = turbidity_level * beta_max_b

    t_r = np.exp(-beta_r * depth_map)
    t_g = np.exp(-beta_g * depth_map)
    t_b = np.exp(-beta_b * depth_map)

    transmission_rgb = np.stack([t_r, t_g, t_b], axis=-1)
    return transmission_rgb


def degrade_image(
    image: np.ndarray,
    turbidity_level: float,
    depth_mode: str = "gradient",
    beta_max_r: float = 3.0,
    beta_max_g: float = 1.8,
    beta_max_b: float = 1.0,
) -> np.ndarray:
    """
    Apply full physically-grounded underwater degradation pipeline to an RGB image.
    """
    if image.dtype == np.uint8:
        img_float = to_float(image)
    else:
        img_float = clip_image(image.copy())

    if turbidity_level <= 0.0:
        return img_float

    # 1. Generate depth map d(x)
    shape = img_float.shape[:2]
    depth_map = generate_depth_map(shape, mode=depth_mode)

    # 2. Compute per-channel transmission map (H, W, 3)
    transmission_rgb = get_per_channel_transmission(
        depth_map,
        turbidity_level,
        beta_max_r=beta_max_r,
        beta_max_g=beta_max_g,
        beta_max_b=beta_max_b,
    )

    # 3. Compute ambient light color vector A (3,)
    ambient_color = get_ambient_light_color(turbidity_level)
    A = ambient_color.reshape(1, 1, 3)

    # 4. Underwater Image Formation Model:
    # I(x) = J(x) * t(x) + A * (1 - t(x))
    degraded_raw = img_float * transmission_rgb + A * (1.0 - transmission_rgb)

    # 5. Apply depth-dependent forward scattering (local contrast reduction / blur)
    degraded_softened = reduce_local_contrast(degraded_raw, turbidity_level, depth_map=depth_map)

    # 6. Add suspended particle noise (marine snow)
    degraded_final = add_particle_noise(degraded_softened, turbidity_level)

    return clip_image(degraded_final)


if __name__ == "__main__":
    print("==================================================")
    print("--- UNDERWATER DEGRADATION DIAGNOSTIC SUITE ---")
    print("==================================================")

    sample_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    sample_filename = "deepfish_00001.jpg"
    sample_path = os.path.join(sample_dir, sample_filename)

    if not os.path.exists(sample_path):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not sample_files:
            print(f"No sample images found in '{sample_dir}'.")
            sys.exit(1)
        sample_path = os.path.join(sample_dir, sample_files[0])

    print(f"Loading sample image: {sample_path}")
    raw_img = load_image(sample_path)

    # 1. Run degrade_image() at THREE turbidity levels: 0.1, 0.5, 0.9
    turbidity_levels = [0.1, 0.5, 0.9]
    print("\n--- GENERATING DEGRADED OUTPUTS ---")
    for turb in turbidity_levels:
        degraded = degrade_image(raw_img, turbidity_level=turb, depth_mode="gradient")
        out_path = os.path.join(output_dir, f"sample_degraded_turb{turb}.png")
        save_image(degraded, out_path)
        print(f"  [+] Saved output at turbidity={turb:.1f} -> '{out_path}'")

    # 2. Ambient light color comparison (0.1 vs 0.9)
    A_01 = get_ambient_light_color(0.1)
    A_09 = get_ambient_light_color(0.9)
    print("\n--- AMBIENT LIGHT COLOR COMPARISON ---")
    print(f"  get_ambient_light_color(0.1): R={A_01[0]:.3f}, G={A_01[1]:.3f}, B={A_01[2]:.3f}")
    print(f"  get_ambient_light_color(0.9): R={A_09[0]:.3f}, G={A_09[1]:.3f}, B={A_09[2]:.3f}")

    # 3. Print diagnostic values for turbidity_level = 0.5 specifically
    print("\n--------------------------------------------------")
    print("--- DIAGNOSTICS FOR TURBIDITY LEVEL = 0.5 ---")
    print("--------------------------------------------------")
    shape = raw_img.shape[:2]
    depth_map = generate_depth_map(shape, mode="gradient")
    turb_test = 0.5
    t_rgb = get_per_channel_transmission(depth_map, turb_test)
    A_test = get_ambient_light_color(turb_test)

    print(f"Depth Map Range       : min = {depth_map.min():.4f}, max = {depth_map.max():.4f}")
    print(f"Transmission Red (R)  : min = {t_rgb[:, :, 0].min():.4f}, max = {t_rgb[:, :, 0].max():.4f}")
    print(f"Transmission Green (G): min = {t_rgb[:, :, 1].min():.4f}, max = {t_rgb[:, :, 1].max():.4f}")
    print(f"Transmission Blue (B) : min = {t_rgb[:, :, 2].min():.4f}, max = {t_rgb[:, :, 2].max():.4f}")
    print(f"Ambient Light Color A : R={A_test[0]:.4f}, G={A_test[1]:.4f}, B={A_test[2]:.4f}")

    print("\nSpatial Transmission Variation (Top Row d=0.0 vs Bottom Row d=1.0):")
    print(f"  Red (R)   : Top mean = {t_rgb[0, :, 0].mean():.4f} | Bottom mean = {t_rgb[-1, :, 0].mean():.4f} (delta = {t_rgb[0, :, 0].mean() - t_rgb[-1, :, 0].mean():.4f})")
    print(f"  Green (G) : Top mean = {t_rgb[0, :, 1].mean():.4f} | Bottom mean = {t_rgb[-1, :, 1].mean():.4f} (delta = {t_rgb[0, :, 1].mean() - t_rgb[-1, :, 1].mean():.4f})")
    print(f"  Blue (B)  : Top mean = {t_rgb[0, :, 2].mean():.4f} | Bottom mean = {t_rgb[-1, :, 2].mean():.4f} (delta = {t_rgb[0, :, 2].mean() - t_rgb[-1, :, 2].mean():.4f})")

    print("==================================================")
