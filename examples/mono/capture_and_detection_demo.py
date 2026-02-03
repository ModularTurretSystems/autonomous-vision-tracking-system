"""
examples/mono/capture_and_detection_demo.py

Author: KrutayaBabka
Date: 2026-01-25 (last modified: 2026-01-30)

Description:
    This script captures frames from a single camera, detects chessboard corners in real time,
    and visualizes the detection results. It can also save frames where the chessboard pattern
    is successfully detected.

    The script is intended for:
        - Verifying camera configuration and focus.
        - Checking chessboard detection quality.
        - Collecting valid calibration images for monocular camera calibration.

    Only frames where the chessboard is successfully detected are saved to disk.

Usage:
    python -m examples.mono.capture_and_detection_demo

Controls:
    - 's' : Save current frame (only if chessboard detected)
    - 'q' : Exit the application
"""


import cv2

from src.calibration.patterns.chessboard import ChessboardPattern
from src.camera.camera import Camera
from src.vision.mono.capture import MonoCapture

from src.utils.image import combine_and_resize_frames


# Camera Configuration
# -------------------------------
CAM_ID = 0  # Camera device index
SAVE_DIR = "data/frames"  # Directory for saving captured frames

PARAMS = (
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720,
)


# -------------------------------
# Chessboard Detection Settings
# -------------------------------
PATTERN_SIZE = (9, 6)  # Number of inner corners per chessboard row and column

WIN_SIZE = (11, 11)    # Window size for subpixel corner refinement
ZERO_ZONE = (-1, -1)
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-3
)

FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FILTER_QUADS
)


# -------------------------------
# Visualization and UI Settings
# -------------------------------
WIN_NAME = "Capture & Detection"

COMBINED_FRAME_SIZE = (1280, 720)  # Resolution of the combined preview window

DELAY = 1  # Delay for cv2.waitKey (ms)
FLIP_CODE = 1  # Horizontal flip for user-friendly preview

EXIT_BUTTON = 'q'
SAVE_BUTTON = 's'

BASE_NAME = "frame"
EXT = "jpg"
SAVE_MESSAGE = "Saved"


# -------------------------------
# Initialize Camera and Pattern
# -------------------------------
cam = Camera(camera_id=CAM_ID, apiPreference=cv2.CAP_MSMF, params=PARAMS)
pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=True, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=CRITERIA)


# -------------------------------
# Initialize Capture Handler
# -------------------------------
capture = MonoCapture(camera=cam, save_dir=SAVE_DIR)
# Alternative initialization:
# capture = MonoCapture(camera=0)


# -------------------------------
# Main Capture Loop
# -------------------------------
while(1):
    # Capture a frame from the camera
    camera_frame = capture.capture_frame()

    # Detect chessboard corners
    res = pattern.detect_corners(camera_frame.frame)

    # Create a copy for visualization
    camera_frame_copy = camera_frame.copy()
    pattern.draw_corners(img=camera_frame_copy.frame, corners=res.corners, patternWasFound=res.found)
    
    # Combine original and annotated frames side by side
    frame = combine_and_resize_frames(frame_size=COMBINED_FRAME_SIZE, frames=[camera_frame.flip(flip_code=FLIP_CODE, in_place=False), camera_frame_copy.flip(flip_code=FLIP_CODE)])

    cv2.imshow(winname=WIN_NAME, mat=frame)

    k = cv2.waitKey(delay=DELAY)

    # Exit application
    if k == ord(EXIT_BUTTON):
        break

    # Save frame only if chessboard is detected
    elif k == ord(SAVE_BUTTON) and res.found:
        if capture.save_frame(frame=camera_frame.frame, base_name=BASE_NAME, ext=EXT): print(SAVE_MESSAGE)

print(f"Number of saved frames = {capture.get_saved_frame_count()}")
