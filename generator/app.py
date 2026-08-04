"""
app.py - Underwater Turbidity Simulator (Professional Classy UI)

An interactive, physically-grounded underwater optical degradation simulator based on the
Jaffe-McGlamery Optical Model and Beer-Lambert Law.

    I(x) = J(x) * t(x) + A * (1 - t(x))
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# Handle imports when run directly or via Streamlit CLI
try:
    from generator.utils import (
        to_float,
        to_uint8,
        clip_image,
        generate_depth_map,
        get_ambient_light_color,
        add_particle_noise,
        reduce_local_contrast,
    )
    from generator.degradation import (
        get_per_channel_transmission,
        degrade_image,
    )
except ImportError:
    from utils import (
        to_float,
        to_uint8,
        clip_image,
        generate_depth_map,
        get_ambient_light_color,
        add_particle_noise,
        reduce_local_contrast,
    )
    from degradation import (
        get_per_channel_transmission,
        degrade_image,
    )

# Page configuration
st.set_page_config(
    page_title="Underwater Turbidity Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Classy Professional Font ('Plus Jakarta Sans'), Custom Gradient Slider & Clean Dark Ocean Theme
CLASSY_PROFESSIONAL_THEME = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0b132b;
        color: #f1f5f9;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #162032;
        border-right: 1px solid #2a374e;
    }
    
    /* Header Titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    .app-header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.3rem;
        letter-spacing: -0.03em;
    }
    
    .app-header-sub {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 0.6rem;
        font-weight: 400;
    }
    
    .app-formula-box {
        font-size: 0.88rem;
        font-family: 'Plus Jakarta Sans', monospace;
        color: #38bdf8;
        background-color: #0f172a;
        padding: 8px 16px;
        border-radius: 6px;
        border: 1px solid #1e293b;
        display: inline-block;
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 1.0rem;
        margin-bottom: 0.75rem;
    }

    .sidebar-section-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    /* Custom Turbidity Gradient Slider Track */
    div[data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #38bdf8 0%, #0284c7 30%, #14b8a6 65%, #78350f 100%) !important;
        height: 10px !important;
        border-radius: 5px !important;
    }
    
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #f43f5e !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 10px rgba(244, 63, 94, 0.7) !important;
        width: 18px !important;
        height: 18px !important;
    }

    .slider-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: -8px;
        margin-bottom: 12px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #111c35;
        border: 1px solid #26354f;
        border-radius: 8px;
        padding: 12px 16px;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }

    /* Bordered Container Cards */
    div[data-testid="stExpander"] {
        border: 1px solid #26354f;
        border-radius: 8px;
        background-color: #111c35;
    }
</style>
"""

st.markdown(CLASSY_PROFESSIONAL_THEME, unsafe_allow_html=True)


def main():
    # Header Section
    st.markdown("<div class='app-header-title'>Underwater Turbidity & Optical Degradation Simulator</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-header-sub'>Physically-grounded simulator based on the Underwater Image Formation Model</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-formula-box'>Physics Model: I(x) = J(x) · t(x) + A · (1 - t(x)) &nbsp;|&nbsp; Beer-Lambert Law: t_c(x) = exp(-β_c · d(x))</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -------------------------------------------------------------------------
    st.sidebar.markdown("<div class='sidebar-section-title'>Simulation Controls</div>", unsafe_allow_html=True)

    # 1. Source Image Selection
    input_source = st.sidebar.radio(
        "Source Selection",
        options=["Select from Dataset", "Upload Custom Image"],
        horizontal=True,
    )

    raw_image = None
    image_name = "sample_image"

    if input_source == "Select from Dataset":
        dataset_choice = st.sidebar.selectbox(
            "Choose Dataset Folder",
            ["data/raw (Fauna)", "data/mangrove_frames (Mangroves)"],
        )
        folder_path = "data/raw" if "raw" in dataset_choice else "data/mangrove_frames"

        if os.path.exists(folder_path):
            valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            files = []
            for ext in valid_exts:
                files.extend(glob.glob(os.path.join(folder_path, ext)))
            files = sorted(list(set(files)))

            if files:
                filenames = [os.path.basename(f) for f in files]
                selected_filename = st.sidebar.selectbox("Select Image", filenames)
                selected_path = os.path.join(folder_path, selected_filename)

                pil_img = Image.open(selected_path).convert("RGB")
                raw_image = np.array(pil_img)
                image_name = selected_filename
            else:
                st.sidebar.warning(f"No image files found in '{folder_path}'.")
        else:
            st.sidebar.error(f"Directory '{folder_path}' does not exist.")

    else:
        uploaded_file = st.sidebar.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            raw_image = np.array(pil_img)
            image_name = uploaded_file.name

    if raw_image is None:
        st.info("Please select or upload a valid source image from the sidebar to begin.")
        return

    # 2. Custom Gradient Water Parameter Controls
    st.sidebar.markdown("<div class='sidebar-section-title'>Water Parameters</div>", unsafe_allow_html=True)

    turbidity_level = st.sidebar.slider(
        "Turbidity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )
    # Custom Gradient Scale Labels
    st.sidebar.markdown(
        """
        <div class="slider-labels">
            <span>Clear (0.0)</span>
            <span>Max (1.0)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    depth_mode = st.sidebar.selectbox(
        "Depth Map Mode",
        options=["gradient", "radial"],
        index=0,
        help="'gradient': Linear top-to-bottom depth profile. 'radial': Center-to-edge distance profile.",
    )

    # 3. Advanced Physical Coefficients
    with st.sidebar.expander("Advanced Physical Coefficients", expanded=False):
        beta_max_r = st.slider("Red Attenuation (beta_max_r)", 1.0, 6.0, 3.0, 0.1)
        beta_max_g = st.slider("Green Attenuation (beta_max_g)", 0.5, 4.0, 1.8, 0.1)
        beta_max_b = st.slider("Blue Attenuation (beta_max_b)", 0.1, 3.0, 1.0, 0.1)

    # -------------------------------------------------------------------------
    # MAIN AREA: VISUAL COMPARISON
    # -------------------------------------------------------------------------
    st.markdown("<div class='section-title'>Visual Comparison</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("##### Clean Original Image")
            st.image(raw_image, use_container_width=True, caption=f"File: {image_name}")

    with col2:
        with st.container(border=True):
            st.markdown(f"##### Degraded Image (Turbidity: {turbidity_level:.2f})")
            
            with st.spinner("Simulating optical degradation..."):
                degraded_float = degrade_image(
                    raw_image,
                    turbidity_level=turbidity_level,
                    depth_mode=depth_mode,
                    beta_max_r=beta_max_r,
                    beta_max_g=beta_max_g,
                    beta_max_b=beta_max_b,
                )
                degraded_uint8 = to_uint8(degraded_float)

            st.image(
                degraded_uint8,
                use_container_width=True,
                caption=f"Depth Mode: '{depth_mode}'",
            )

    # -------------------------------------------------------------------------
    # PHYSICAL DIAGNOSTICS & METRICS
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("<div class='section-title'>Optical Diagnostics & Parameters</div>", unsafe_allow_html=True)

    h, w = raw_image.shape[:2]
    depth_map = generate_depth_map((h, w), mode=depth_mode)
    t_rgb = get_per_channel_transmission(
        depth_map,
        turbidity_level,
        beta_max_r=beta_max_r,
        beta_max_g=beta_max_g,
        beta_max_b=beta_max_b,
    )
    A = get_ambient_light_color(turbidity_level)

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)

    with d_col1:
        st.metric(
            label="Ambient Color A (RGB)",
            value=f"({A[0]:.2f}, {A[1]:.2f}, {A[2]:.2f})",
        )
        ambient_box = np.zeros((24, 150, 3), dtype=np.uint8)
        ambient_box[:] = (A * 255).astype(np.uint8)
        st.image(ambient_box, caption="Backscatter Tint", use_container_width=True)

    with d_col2:
        st.metric(
            label="Red (R) Transmission Range",
            value=f"{t_rgb[:,:,0].min():.2f} - {t_rgb[:,:,0].max():.2f}",
            delta=f"-{(t_rgb[0,:,0].mean() - t_rgb[-1,:,0].mean()):.2f} delta",
        )

    with d_col3:
        st.metric(
            label="Green (G) Transmission Range",
            value=f"{t_rgb[:,:,1].min():.2f} - {t_rgb[:,:,1].max():.2f}",
            delta=f"-{(t_rgb[0,:,1].mean() - t_rgb[-1,:,1].mean()):.2f} delta",
        )

    with d_col4:
        st.metric(
            label="Blue (B) Transmission Range",
            value=f"{t_rgb[:,:,2].min():.2f} - {t_rgb[:,:,2].max():.2f}",
            delta=f"-{(t_rgb[0,:,2].mean() - t_rgb[-1,:,2].mean()):.2f} delta",
        )

    # Minimum Transmission Bar Chart
    st.markdown("##### Minimum Transmission Spectrum (Maximum Depth Attenuation)")
    chart_data = pd.DataFrame(
        {
            "Minimum Transmission": [
                t_rgb[:, :, 0].min(),
                t_rgb[:, :, 1].min(),
                t_rgb[:, :, 2].min(),
            ]
        },
        index=["Red Channel (~700nm)", "Green Channel (~530nm)", "Blue Channel (~470nm)"],
    )
    st.bar_chart(chart_data, height=180)


if __name__ == "__main__":
    main()
