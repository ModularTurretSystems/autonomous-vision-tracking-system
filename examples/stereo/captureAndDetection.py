"""
src/examples/stereo/captureAndDetection.py

Author: KrutayaBabka
Date: 2026-01-23 (last modified: 2026-01-29)

Description:
    This script captures synchronized frames from a stereo camera system (left and right cameras),
    detects chessboard corners on both images in real time, and visualizes the detection results.

    It is primarily intended for:
        - Verifying correct stereo camera alignment and configuration.
        - Collecting valid stereo calibration image pairs.
        - Providing visual feedback for chessboard detection quality.

    The script allows the user to:
        - View detected chessboard corners on both camera streams.
        - Save stereo frame pairs only when the pattern is successfully detected in both images.

Usage:
    python -m examples.stereo.captureAndDetection

Controls:
    - 's' : Save current stereo frame pair (only if chessboard detected in both images)
    - 'q' : Exit the application
"""


import cv2

from src.camera.camera import Camera
from src.vision.stereo.system import StereoSystem
from src.vision.stereo.capture import StereoCapture
from src.calibration.patterns.chessboard import ChessboardPattern

from src.utils.image import combine_and_resize_frames


# -------------------------------
# Camera Configuration
# -------------------------------
CAM_L_ID = 1  # Left camera device index
CAM_R_ID = 2  # Right camera device index

# Camera capture parameters (applied to both cameras)
PARAMS = (
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720,
)

SAVE_DIR = "data/calibration_images" # Directory for saving stereo image pairs

# -------------------------------
# Chessboard Detection Settings
# -------------------------------
PATTERN_SIZE = (9, 6) # Number of inner corners per chessboard row and column

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

# Subpixel corner refinement parameters
WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-3
)

# -------------------------------
# Visualization Settings
# -------------------------------
COMBINED_RESOLUTION = (1280, 720) # Resolution of the combined stereo preview window

BASE_NAME = "frame"   # Base filename for saved images
EXT = "png"           # Image format
WINNAME = "Result"    # OpenCV window name
DELAY = 1             # Delay for cv2.waitKey (ms)
FLIP_CODE = 1         # Horizontal flip (mirror view for user convenience)

EXIT_BUTTON = 'q'
SAVE_BUTTON = 's'
SAVE_MESSAGE = "Saved"


# -------------------------------
# Initialize Cameras and Stereo System
# -------------------------------
cam_l = Camera(camera_id=CAM_L_ID, apiPreference=cv2.CAP_MSMF, params=PARAMS)
cam_r = Camera(camera_id=CAM_R_ID, apiPreference=cv2.CAP_MSMF, params=PARAMS)
stereo_system = StereoSystem(left=cam_l, right=cam_r)
stereo_capture = StereoCapture(stereo_system=stereo_system, save_dir=SAVE_DIR)


# -------------------------------
# Initialize Chessboard Pattern
# -------------------------------
pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=True, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=CRITERIA)


# -------------------------------
# Main Capture Loop
# -------------------------------
while(1):
    # Capture synchronized stereo frame
    stereo_frame = stereo_capture.capture_frame()
    frame_l = stereo_frame.camera_frame_l.frame
    frame_r = stereo_frame.camera_frame_r.frame

    # Detect chessboard corners in both images
    res_l = pattern.detect_corners(img=frame_l)
    res_r = pattern.detect_corners(img=frame_r)

    # Create a copy of the frame for visualization
    stereo_frame_copy = stereo_frame.copy()

    # Draw detected corners
    pattern.draw_corners(img=stereo_frame_copy.camera_frame_l.frame, corners=res_l.corners, patternWasFound=res_l.found)
    pattern.draw_corners(img=stereo_frame_copy.camera_frame_r.frame, corners=res_r.corners, patternWasFound=res_r.found)

    # Flip images for user-friendly preview
    stereo_frame_copy.flip(flip_code=FLIP_CODE)

    # Combine left and right frames into a single display image
    frame = combine_and_resize_frames(frame_size=COMBINED_RESOLUTION, frames=stereo_frame_copy.to_list(), horizontal=True)

    cv2.imshow(winname=WINNAME, mat=frame)

    k = cv2.waitKey(delay=DELAY)

    # Exit application
    if k == ord(EXIT_BUTTON):
        break
    
    # Save frame pair only if chessboard is detected in both images
    elif k == ord(SAVE_BUTTON) and res_l.found and res_r.found:
        if stereo_capture.save_frame(frame=stereo_frame, base_name=BASE_NAME, ext=EXT): print(SAVE_MESSAGE)

print(f"Number of saved frames = {stereo_capture.get_saved_frame_count()}")
