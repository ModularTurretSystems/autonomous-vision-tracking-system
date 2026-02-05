"""
examples/extract_corners_from_directory_demo.py

Author: KrutayaBabka
Date: 2026-01-23

Description:
    This script demonstrates how to extract chessboard corners from a directory of images
    using OpenCV. The extracted features can be used for camera calibration. It utilizes
    the DirectoryCalibrationDataset and ChessboardPattern classes for convenient processing.

    The script prints the total number of successfully detected feature sets.

Usage:
    python -m examples.extract_corners_from_directory_demo
"""


import cv2

from src.calibration.dataset.directory import DirectoryCalibrationDataset
from src.calibration.patterns.chessboard import ChessboardPattern


# ===============================
# Constants / Configuration
# ===============================
DATA_DIR = "data/frames/"  # Path to directory containing calibration images

# Chessboard pattern settings
PATTERN_SIZE = (9, 6)      # Number of inner corners per chessboard row and column
FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FILTER_QUADS
)
REFINE_CORNERS = True       # Enable subpixel refinement
WIN_SIZE = (11, 11)         # Window size for corner refinement
ZERO_ZONE = (-1, -1)        # Zero zone for corner refinement
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.01
)


# ===============================
# Initialize Chessboard Pattern
# ===============================
pattern = ChessboardPattern(
    pattern_size=PATTERN_SIZE,
    flags=FLAGS,
    refine=REFINE_CORNERS,
    win_size=WIN_SIZE,
    zero_zone=ZERO_ZONE,
    criteria=SUBPIX_CRITERIA
)


# ===============================
# Load Dataset and Extract Features
# ===============================
dataset = DirectoryCalibrationDataset(data_dir=DATA_DIR, pattern=pattern)

# Extract chessboard corners from all images in the directory
features = dataset.extract_features()

# Print total number of successful detections
print(f"Number of images with detected corners: {len(features)}")
