"""
utils.py - Physical Underwater Degradation Helper Functions

This module provides helper utilities for simulating physical underwater optical degradation
based on the classic Underwater Image Formation Model (Jaffe-McGlamery & Beer-Lambert law):

    I(x) = J(x) * t(x) + A * (1 - t(x))

Physics Principles Covered:
---------------------------
1. Beer-Lambert Law: Light intensity exponentially decays as it travels through water.
2. Light Attenuation & Scattering: Water selectively absorbs and scatters light.
3. Ambient Backscatter: Suspended particles scatter background ambient light into the camera line of sight.
4. Forward Scattering: Multiple small-angle scattering events blur image details and reduce local contrast.
5. Suspended Sediment (Marine Snow): Random particles scatter direct illumination, producing bright specks.
"""

from __future__ import annotations

import os
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk in RGB format.

    OpenCV reads images in BGR color space by default. This function converts BGR -> RGB
    so all internal processing occurs in standard RGB color space.

    Args:
        path (str): File path to the image.

    Returns:
        np.ndarray: Loaded image array in RGB format (uint8, shape [H, W, 3]).

    Raises:
        FileNotFoundError: If the image cannot be read or path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path does not exist: '{path}'")

    image_bgr = cv2.imread(path)
    if image_bgr is None:
        raise ValueError(f"Unable to decode image from path: '{path}'")

    # Convert BGR (OpenCV default) to RGB (Standard representation)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save an RGB image (uint8 or float) to disk.

    Converts float images [0, 1] to uint8 [0, 255] and handles RGB -> BGR conversion
    for OpenCV writing. Creates parent output directories if they do not exist.

    Args:
        image (np.ndarray): Image array in RGB space (uint8 or float64).
        path (str): Destination file path.
    """
    # Ensure directory exists
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Convert float image [0, 1] to uint8 [0, 255]
    if image.dtype != np.uint8:
        image_uint8 = to_uint8(image)
    else:
        image_uint8 = image

    # Convert RGB -> BGR for OpenCV
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def to_float(image: np.ndarray) -> np.ndarray:
    """
    Convert a uint8 image [0, 255] to a float64 representation normalized to [0.0, 1.0].

    Physical computations (multiplication by transmission maps, adding ambient light)
    require floating-point precision to avoid integer overflow/underflow artifacts.

    Args:
        image (np.ndarray): Image array with uint8 values in range [0, 255].

    Returns:
        np.ndarray: Image array with float64 values in range [0.0, 1.0].
    """
    return image.astype(np.float64) / 255.0


def clip_image(image: np.ndarray) -> np.ndarray:
    """
    Clip floating-point image values strictly to the valid range [0.0, 1.0].

    Prevents out-of-bounds values caused by additive noise or extreme contrast adjustments.

    Args:
        image (np.ndarray): Floating-point image array.

    Returns:
        np.ndarray: Clipped floating-point image array in range [0.0, 1.0].
    """
    return np.clip(image, 0.0, 1.0)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert a float image [0.0, 1.0] back to uint8 representation [0, 255].

    Clips out-of-range values before scaling and casting to uint8 integers.

    Args:
        image (np.ndarray): Floating-point image array.

    Returns:
        np.ndarray: Image array with uint8 values in range [0, 255].
    """
    clipped = clip_image(image)
    return (clipped * 255.0).round().astype(np.uint8)


def generate_depth_map(shape: tuple, mode: str = "gradient") -> np.ndarray:
    """
    Generate a synthetic normalized depth map d(x) in range [0.0, 1.0].

    Physical Reasoning:
    -------------------
    Light attenuation in water depends directly on distance (depth/range from camera).
    In real scenes, far objects suffer more degradation than near objects.
    - 'gradient': Models a typical underwater perspective (top/near surface to bottom/far distance).
    - 'radial': Models a central subject closer to the camera with surrounding water receding into distance.

    Args:
        shape (tuple): Spatial dimensions of the target image (height, width).
        mode (str): 'gradient' (top-to-bottom depth profile) or 'radial' (center-to-edge profile).

    Returns:
        np.ndarray: 2D depth map array of shape (H, W) normalized between 0.0 (near) and 1.0 (far).
    """
    height, width = shape[:2]

    if mode == "gradient":
        # Linear vertical gradient: top = 0.0 (near), bottom = 1.0 (farther depth)
        y_coords = np.linspace(0.0, 1.0, height, dtype=np.float64)
        depth_map = np.tile(y_coords[:, np.newaxis], (1, width))

    elif mode == "radial":
        # Distance from image center normalized so corners are 1.0
        center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        depth_map = dist_from_center / max_dist

    else:
        raise ValueError(f"Unknown depth mode '{mode}'. Choose 'gradient' or 'radial'.")

    return clip_image(depth_map)


def compute_transmission_map(depth_map: np.ndarray, turbidity_level: float, beta_max: float = 3.0) -> np.ndarray:
    """
    Compute scalar transmission map t(x) using Beer-Lambert's Exponential Law:

        t(x) = exp(-beta * d(x))

    Physical Reasoning:
    -------------------
    As light travels through turbid water of depth d(x), its intensity exponentially decreases.
    beta represents the medium extinction/attenuation coefficient, scaled linearly by turbidity_level.

    Args:
        depth_map (np.ndarray): Normalized depth map [0.0, 1.0] of shape (H, W).
        turbidity_level (float): Turbidity factor in range [0.0, 1.0] (0 = crystal clear, 1 = max turbid).
        beta_max (float): Maximum attenuation coefficient for full turbidity.

    Returns:
        np.ndarray: Transmission map t(x) of shape (H, W) with values in (0.0, 1.0].
    """
    beta = turbidity_level * beta_max
    transmission = np.exp(-beta * depth_map)
    return transmission


def get_ambient_light_color(turbidity_level: float) -> np.ndarray:
    """
    Calculate the ambient backscatter light color vector A = [R, G, B] in range [0.0, 1.0].

    Physical Reasoning:
    -------------------
    Water bodies scatter ambient sunlight back into the camera lens (backscatter haze).
    - Clear water exhibits a light neutral cyan-blue ambient light [0.38, 0.52, 0.58].
    - Highly turbid/murky water contains suspended sediment, silt, and algae, producing a
      desaturated, earthy greenish-gray/teal haze [0.22, 0.42, 0.32].

    Args:
        turbidity_level (float): Turbidity factor in range [0.0, 1.0].

    Returns:
        np.ndarray: 1D RGB float array [R, G, B] representing ambient light color.
    """
    # Clear water ambient light (light neutral cyan-blue)
    clear_ambient = np.array([0.38, 0.52, 0.58], dtype=np.float64)

    # Desaturated murky/turbid water ambient light (earthy greenish-gray teal)
    murky_ambient = np.array([0.22, 0.42, 0.32], dtype=np.float64)

    # Linear interpolation based on turbidity level
    ambient = (1.0 - turbidity_level) * clear_ambient + turbidity_level * murky_ambient
    return ambient


def add_particle_noise(
    image: np.ndarray,
    turbidity_level: float,
    max_density: float = 0.02,
    max_intensity: float = 0.6
) -> np.ndarray:
    """
    Add random bright specks simulating marine snow and suspended particles.

    Physical Reasoning:
    -------------------
    Turbid water contains suspended organic matter, sand, and plankton ("marine snow").
    These particles reflect ambient light directly into the camera lens, creating bright,
    out-of-focus specks whose quantity and brightness increase with water turbidity.

    Args:
        image (np.ndarray): Input float image in RGB format [0.0, 1.0].
        turbidity_level (float): Turbidity factor in range [0.0, 1.0].
        max_density (float): Maximum fraction of pixels containing particle specks at full turbidity.
        max_intensity (float): Maximum brightness added by particles.

    Returns:
        np.ndarray: Degraded float image with particle specks added.
    """
    if turbidity_level <= 0.0:
        return image.copy()

    height, width = image.shape[:2]

    # Density and intensity scale linearly with turbidity
    current_density = turbidity_level * max_density
    current_intensity = turbidity_level * max_intensity

    # Generate random mask for particle placement
    random_mask = np.random.rand(height, width)
    particle_locs = random_mask < current_density

    # Generate random brightness for each speck
    speck_brightness = np.random.uniform(0.3, 1.0, size=(height, width, 1)) * current_intensity

    # Create particle overlay image
    particle_overlay = np.zeros_like(image)
    particle_overlay[particle_locs] = speck_brightness[particle_locs]

    # Add specks to original image and clip
    noisy_image = image + particle_overlay
    return clip_image(noisy_image)


def reduce_local_contrast(
    image: np.ndarray,
    turbidity_level: float,
    depth_map: np.ndarray | None = None,
    blur_ksize: int = 25,
    max_blend: float = 0.5
) -> np.ndarray:
    """
    Reduce local image contrast and blur fine details to model forward scattering.

    Physical Reasoning:
    -------------------
    Forward scattering diffuses light and reduces contrast progressively with depth d(x).
    Near objects (d(x) ~ 0) experience negligible forward scattering blur, whereas far
    objects (d(x) ~ 1) experience significant blurring and contrast degradation.

    Args:
        image (np.ndarray): Input float image in RGB format [0.0, 1.0].
        turbidity_level (float): Turbidity factor in range [0.0, 1.0].
        depth_map (np.ndarray, optional): Normalized depth map [0.0, 1.0].
        blur_ksize (int): Kernel size for Gaussian blur (must be an odd integer).
        max_blend (float): Maximum blend ratio of blurred image at full turbidity.

    Returns:
        np.ndarray: Image with reduced local contrast and softened edges.
    """
    if turbidity_level <= 0.0:
        return image.copy()

    # Ensure kernel size is odd
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    # Gaussian blur to simulate forward-scattered diffuse light
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    # Compute depth-dependent blend factor alpha(x)
    if depth_map is not None:
        alpha = (turbidity_level * depth_map * max_blend)[:, :, np.newaxis]
    else:
        alpha = turbidity_level * max_blend

    softened_image = (1.0 - alpha) * image + alpha * blurred
    return clip_image(softened_image)
