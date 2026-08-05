"""
extract_frames.py
------------------
A simple, reusable command-line tool that extracts frames from a video
(.mp4) at a fixed time interval, and saves them as numbered .jpg images.

WHY THIS SCRIPT EXISTS
-----------------------
For this project, the mangrove frames were actually collected from YouTube
dive videos and annotated collaboratively in Roboflow, so this script was
not needed to produce data/mangrove_frames/. It's kept here as a clean,
documented, reusable tool in case raw video needs to be turned into frames
again later (new footage, another dataset, etc.).

HOW IT WORKS (high level)
--------------------------
1. Open the video with OpenCV (cv2.VideoCapture).
2. Ask OpenCV for the video's FPS (frames per second) and total frame count.
3. Convert "extract 1 frame every N seconds" into "extract 1 frame every
   X video frames" by multiplying N seconds x FPS.
4. Walk through the video frame by frame. Every time we hit a frame that
   is a multiple of that interval, we save it as a .jpg image.
5. Show a tqdm progress bar while this happens.
6. Print a short summary at the end.

USAGE EXAMPLES
---------------
    # Extract 1 frame every 5 seconds (default) into ./frames_out
    python extract_frames.py my_video.mp4 frames_out

    # Extract 1 frame every 2 seconds
    python extract_frames.py my_video.mp4 frames_out --interval 2

OUTPUT
------
Frames are saved into the given output folder as:
    frame_0001.jpg
    frame_0002.jpg
    frame_0003.jpg
    ...
"""

# ---- Standard library imports -------------------------------------------
import argparse                # for reading command-line arguments
from pathlib import Path       # for clean, cross-platform file paths

# ---- Third-party imports --------------------------------------------------
import cv2                     # OpenCV: reads and decodes the video
from tqdm import tqdm          # tqdm: shows a nice progress bar


def parse_arguments():
    """
    Reads the command-line arguments the user typed in, e.g.:
        python extract_frames.py video.mp4 out_folder --interval 5

    Returns an object with .video_path, .output_folder, .interval
    """
    parser = argparse.ArgumentParser(
        description="Extract frames from a video every N seconds."
    )

    # Required: path to the input video file
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input .mp4 video file."
    )

    # Required: path to the folder where frames will be saved
    parser.add_argument(
        "output_folder",
        type=str,
        help="Folder where extracted frames will be saved."
    )

    # Optional: how many seconds between each extracted frame
    parser.add_argument(
        "--interval",
        type=float,
        default=5,
        help="Extract 1 frame every N seconds (default: 5)."
    )

    return parser.parse_args()


def extract_frames(video_path: str, output_folder: str, interval_seconds: float):
    """
    Extracts frames from `video_path` every `interval_seconds` seconds
    and saves them into `output_folder`.
    """

    # --- Step 1: Make sure the input video actually exists -----------------
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # --- Step 2: Make sure the output folder exists -------------------------
    # parents=True  -> also creates any missing parent folders
    # exist_ok=True -> don't error if the folder is already there
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- Step 3: Open the video with OpenCV ---------------------------------
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    # --- Step 4: Read basic video info --------------------------------------
    fps = video.get(cv2.CAP_PROP_FPS)                      # frames per second
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # total frame count

    if fps <= 0:
        # Some video files/codecs don't report FPS correctly.
        # Fall back to a safe default so we don't divide by zero later.
        print("Warning: could not read FPS from video, defaulting to 25.")
        fps = 25.0

    duration_seconds = total_frames / fps

    # --- Step 5: Work out how many video frames = "interval_seconds" -------
    # Example: fps = 30, interval_seconds = 5  ->  frame_interval = 150
    # That means we keep 1 out of every 150 frames.
    frame_interval = max(1, int(round(fps * interval_seconds)))

    # --- Step 6: Loop through the video and save the frames we want --------
    saved_count = 0        # how many frames we've saved so far
    current_frame_index = 0  # which frame number we're currently reading

    # tqdm wraps our loop and prints a live progress bar.
    # total=total_frames tells it how far along we are (0 -> 100%).
    with tqdm(total=total_frames, desc="Reading video", unit="frame") as progress_bar:
        while True:
            success, frame = video.read()  # read the next frame

            if not success:
                # No more frames left -> end of video
                break

            # Only save a frame if it lands on our interval
            if current_frame_index % frame_interval == 0:
                saved_count += 1
                # Zero-padded 4-digit filename: frame_0001.jpg, frame_0002.jpg, ...
                filename = f"frame_{saved_count:04d}.jpg"
                filepath = output_folder / filename

                # Write the frame to disk as a .jpg image
                cv2.imwrite(str(filepath), frame)

            current_frame_index += 1
            progress_bar.update(1)

    # --- Step 7: Clean up ----------------------------------------------------
    video.release()

    # --- Step 8: Print a summary ---------------------------------------------
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)

    print("\n--- Extraction Summary ---")
    print(f"Video file:        {video_path.name}")
    print(f"Video duration:    {minutes} min {seconds} sec ({duration_seconds:.1f} sec total)")
    print(f"Video FPS:         {fps:.2f}")
    print(f"Frame interval:    every {interval_seconds} sec (~every {frame_interval} frames)")
    print(f"Frames extracted:  {saved_count}")
    print(f"Saved to folder:   {output_folder.resolve()}")
    print("--------------------------\n")

    return saved_count


def main():
    args = parse_arguments()
    extract_frames(
        video_path=args.video_path,
        output_folder=args.output_folder,
        interval_seconds=args.interval,
    )


# This makes sure main() only runs when the script is executed directly,
# e.g. `python extract_frames.py ...`, and not when imported elsewhere.
if __name__ == "__main__":
    main()
