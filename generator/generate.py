"""
generate.py

CLI interface to run turbidity degradation on a folder of images.
"""

import argparse
import os
import cv2
from degradation import apply_turbidity

def main():
    parser = argparse.ArgumentParser(description="CLI to degrade images with turbidity simulation.")
    parser.add_argument("--input", required=True, help="Input directory of raw images")
    parser.add_argument("--output", required=True, help="Output directory for degraded images")
    parser.add_argument("--turbidity", type=float, default=0.5, help="Turbidity level (0.0 to 1.0)")
    
    args = parser.parse_args()
    print(f"Processing images from {args.input} to {args.output} at turbidity level {args.turbidity}...")

if __name__ == "__main__":
    main()
