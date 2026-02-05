"""
examples/chessboard_corners_detection_demo.py

Author: KrutayaBabka
Date: 2026-01-23

Description:
    This script demonstrates real-time detection and visualization of chessboard corners
    from a live camera stream. It uses a ChessboardPattern class for corner detection
    and highlights detected corners on the original image. The result is displayed
    side by side with the original frame for easy comparison.

    The script allows customization of:
        - Chessboard pattern size and subpixel refinement
        - Camera capture resolution
        - Display resolution and layout
        - Exit key for closing the application

Usage:
    python -m examples.chessboard_corners_detection_demo

Controls:
    - EXIT_BUTTON : Exit the application
"""


import cv2

from src.utils.image import combine_and_resize_frames
from src.camera.camera import Camera
from src.calibration.patterns.chessboard import ChessboardPattern


# ===============================
# Constants / Configuration
# ===============================
# Camera settings
CAMERA_ID = 0
CAMERA_PARAMS = (
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720,
)

# Chessboard pattern settings
PATTERN_SIZE = (9, 6)          # Number of inner corners per chessboard row and column
WIN_SIZE = (11, 11)             # Window size for subpixel refinement
ZERO_ZONE = (-1, -1)
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.01
)
FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FILTER_QUADS
)

# Display settings
SCREEN_RESOLUTION = (1280, 720)
ROWS, COLS = 2, 2              # Layout for combined display
DELAY = 1                       # Delay for cv2.waitKey (ms)
EXIT_BUTTON = 'q'               # Key to exit the application
WINDOW_NAME = "Chessboard Corners"


# ===============================
# Initialize Camera and Pattern
# ===============================
cam = Camera(camera_id=CAMERA_ID, apiPreference=cv2.CAP_MSMF, params=CAMERA_PARAMS)

pattern = ChessboardPattern(
    pattern_size=PATTERN_SIZE,
    flags=FLAGS,
    refine=True,
    win_size=WIN_SIZE,
    zero_zone=ZERO_ZONE,
    criteria=CRITERIA
)

# Compute size of each frame in the combined display
img_w = SCREEN_RESOLUTION[0] // COLS
img_h = SCREEN_RESOLUTION[1] // ROWS


# ===============================
# Main Detection Loop
# ===============================
while True:
    # Capture a frame from the camera
    camera_frame = cam.capture_frame()
    frame = camera_frame.frame

    # Detect chessboard corners
    res = pattern.detect_corners(img=frame)

    # Draw detected corners on a copy
    frame_with_corners = frame.copy()
    pattern.draw_corners(img=frame_with_corners, corners=res.corners, patternWasFound=res.found)

    # Flip the annotated frame horizontally for user-friendly preview
    cv2.flip(src=frame_with_corners, flipCode=1, dst=frame_with_corners)

    # Combine original and annotated frames side by side
    res_frame = combine_and_resize_frames(
        frame_size=SCREEN_RESOLUTION,
        frames=[frame, frame_with_corners],
        horizontal=True
    )

    # Display the combined frame
    cv2.imshow(winname=WINDOW_NAME, mat=res_frame)

    # Exit on key press
    if cv2.waitKey(delay=DELAY) == ord(EXIT_BUTTON):
        break

cv2.destroyAllWindows()
