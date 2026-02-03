"""
examples/charuco_create_demo.py

Author: KrutayaBabka
Date: 2026-01-23

Description:
    This script demonstrates how to create and display a Charuco board image using OpenCV's
    ArUco module. Charuco boards are used for camera calibration and pose estimation,
    combining ArUco markers with a chessboard pattern for higher accuracy.

    The script allows customization of:
        - The board size (number of squares in rows and columns)
        - The square and marker sizes
        - The output image resolution and margins
        - The display delay before closing the window

Usage:
    python -m examples.charuco_create_demo
"""


import cv2
from cv2 import aruco
from cv2.aruco import CharucoBoard


# ===============================
# Constants / Configuration
# ===============================
BOARD_ROWS = 5                 # Number of squares in vertical direction
BOARD_COLS = 7                 # Number of squares in horizontal direction
SQUARE_LENGTH = 0.03           # Square side length in meters
MARKER_LENGTH = 0.015          # Marker side length in meters

DICT_TYPE = aruco.DICT_4X4_50  # Predefined ArUco dictionary

OUT_IMAGE_SIZE = (1366, 768)   # Output image resolution in pixels (width, height)
MARGIN_SIZE = 10                # Margin around the board in pixels
BORDER_BITS = 1                 # Width of marker border bits in pixels

WINDOW_NAME = "Charuco Board"
DELAY = 0                        # Delay for cv2.waitKey (ms), 0 waits indefinitely


# ===============================
# Create Charuco Board
# ===============================
# Initialize Charuco board with given parameters
board = CharucoBoard(
    size=(BOARD_ROWS, BOARD_COLS),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=aruco.getPredefinedDictionary(DICT_TYPE)
)

# Generate the board image
board_image = board.generateImage(
    outSize=OUT_IMAGE_SIZE,
    marginSize=MARGIN_SIZE,
    borderBits=BORDER_BITS
)


# ===============================
# Display the Board
# ===============================
cv2.imshow(winname=WINDOW_NAME, mat=board_image)
cv2.waitKey(delay=DELAY)
cv2.destroyAllWindows()
