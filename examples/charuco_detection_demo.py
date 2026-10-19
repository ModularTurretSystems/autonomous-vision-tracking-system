"""
examples/charuco_detection_demo.py

Author: KrutayaBabka
Date: 2026-01-23

Description:
    This script demonstrates real-time detection of a Charuco board from a live camera
    stream. It uses OpenCV's ArUco module for marker detection and a CharucoPattern
    wrapper for convenience. The detected board is visualized in real time.

    The script allows customization of:
        - Charuco board size and marker dimensions
        - Camera capture resolution
        - Detector and refinement parameters

Usage:
    python -m examples.charuco_detection_demo

Controls:
    - EXIT_BUTTON : Exit the application
"""


import cv2
from cv2 import aruco
from cv2.aruco import CharucoBoard, getPredefinedDictionary

from src.camera.camera import Camera
from src.calibration.patterns.charuco import CharucoPattern


# ===============================
# Constants / Configuration
# ===============================
# Charuco board configuration
BOARD_SIZE = (5, 7)             # Number of squares (rows, columns)
SQUARE_LENGTH = 0.03            # Square side length in meters
MARKER_LENGTH = 0.015           # Marker side length in meters
ARUCO_DICTIONARY = aruco.DICT_5X5_100

# Camera capture configuration
CAMERA_ID = 0
CAMERA_PARAMS = [
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720
]
DISPLAY_RESOLUTION = (1280, 720)

# Detector / Display parameters
DETECTOR_DELAY = 1              # Delay for cv2.waitKey in ms
EXIT_BUTTON = 'q'               # Key to exit the application
WINDOW_NAME = "Charuco Detection"


# ===============================
# Initialize Charuco Board
# ===============================
dictionary = getPredefinedDictionary(dict=ARUCO_DICTIONARY)

board = CharucoBoard(
    size=BOARD_SIZE,
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=dictionary
)

# Refinement and detection parameters
refine_params = aruco.RefineParameters(checkAllOrders=True)
detector_params = aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

# Initialize CharucoPattern helper
pattern = CharucoPattern(
    charuco_board=board,
    detector_params=detector_params,
    refine_params=refine_params
)


# ===============================
# Initialize Camera
# ===============================
cam = Camera(camera_id=CAMERA_ID, apiPreference=cv2.CAP_V4L2, params=CAMERA_PARAMS)


# ===============================
# Main Detection Loop
# ===============================
while True:
    # Capture a frame from the camera
    camera_frame = cam.capture_frame()
    frame = camera_frame.frame

    # Detect Charuco board
    res = pattern.detect_charuco_board(img=frame)

    # Draw detected board and markers
    pattern.draw_charuco_board(img=frame, res=res)
    # Optional alternatives:
    # pattern.draw_corners(img=frame, res=res)
    # pattern.draw_charuco_board_manual(img=frame, marker_corners=res.markers.corners)

    # Display the frame
    cv2.imshow(winname=WINDOW_NAME, mat=frame)

    # Exit on key press
    if cv2.waitKey(delay=DETECTOR_DELAY) == ord(EXIT_BUTTON):
        break

cv2.destroyAllWindows()
