"""
examples/mono/mono_calibration_demo.py

Author: KrutayaBabka
Date: 2026-01-29

Description: 
    This script demonstrates the usage of the MonoCalibrator class to calibrate a monocular camera
    using a chessboard pattern. It extracts chessboard corners from a dataset of images and performs
    camera calibration, returning intrinsic and extrinsic parameters along with distortion coefficients.

Attributes in the calibration result (calibration_result):
- rms: Root Mean Square reprojection error
- camera_matrix: Intrinsic camera matrix
- dist_coeffs: Distortion coefficients
- rvecs: Rotation vectors for each calibration image
- tvecs: Translation vectors for each calibration image
- object_points: 3D object points of the pattern
- std_intrinsics: Standard deviations of intrinsic parameters
- std_extrinsics: Standard deviations of extrinsic parameters
- std_object_points: Standard deviations of object points
- per_view_error: Reprojection error per image

Usage:
    python -m examples.mono.mono_calibration_demo
"""


import cv2

from src.calibration.patterns.chessboard import ChessboardPattern
from src.calibration.dataset.directory import DirectoryCalibrationDataset
from src.calibration.mono.calibrator import MonoCalibrator
from src.calibration.mono.constants import FixedPointMode


# -------------------------------
# Calibration Pattern Settings
# -------------------------------
PATTERN_SIZE = (9, 6)  # Number of inner corners per chessboard row and column
SQUARE_SIZE = 30.0     # Size of a square in mm

# Flags for chessboard corner detection
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
# Optional additional flags examples:
# FLAGS |= cv2.CALIB_CB_FILTER_QUADS
# FLAGS |= cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ACCURACY

REFINE_CORNERS = True
WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# -------------------------------
# Dataset and Image Settings
# -------------------------------
DATA_DIR = "data/frames"
IMAGE_SIZE = (1280, 720)

# Calibration flags
CALIBRATOR_FLAGS = None
# Optional examples:
# CALIBRATOR_FLAGS = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
# CALIBRATOR_FLAGS |= cv2.CALIB_USE_INTRINSIC_GUESS

# Termination criteria for calibration optimization
CALIBRATOR_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# -------------------------------
# Initialize Calibration Pattern
# -------------------------------
pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=REFINE_CORNERS, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=SUBPIX_CRITERIA, square_size=SQUARE_SIZE)

# -------------------------------
# Load Dataset and Extract Features
# -------------------------------
data_set = DirectoryCalibrationDataset(data_dir=DATA_DIR, pattern=pattern)
image_points  = data_set.extract_features()

# -------------------------------
# Initialize Calibrator and Run Calibration
# -------------------------------
calibrator = MonoCalibrator(pattern=pattern, image_size=IMAGE_SIZE, image_points=image_points, i_fixed_point=FixedPointMode.TOP_RIGHT, flags=CALIBRATOR_FLAGS, criteria=CALIBRATOR_CRITERIA)

# Run calibration
calibration_result = calibrator.calibrate()

# -------------------------------
# Display Calibration Results
# -------------------------------
print("=== Calibration Results ===")
print(f"RMS Reprojection Error: {calibration_result.rms:.6f}", '\n')
print("Camera Matrix:\n", calibration_result.camera_matrix, '\n')
print("Distortion Coefficients:\n", calibration_result.dist_coeffs, '\n')
print("Per-view Reprojection Error:\n", calibration_result.per_view_error, '\n')
