"""
examples/stereo/stereoCalibration.py

Author: KrutayaBabka
Date: 2026-01-30

Description:
    This script demonstrates how to perform calibration for a stereo camera system using
    a chessboard pattern. It extracts chessboard corners from separate datasets for the
    left and right cameras, performs monocular calibration for each camera individually,
    and then displays a comparison of the calibration results.

    The script is intended for:
        - Computing intrinsic parameters (camera matrix, distortion coefficients) for each camera
        - Comparing left and right camera calibration results
        - Preparing data for stereo calibration pipelines

Calibration Result Attributes:
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
    python -m examples.stereo.stereoCalibration
"""


import cv2

from src.calibration.patterns.chessboard import ChessboardPattern
from src.calibration.dataset.directory import DirectoryCalibrationDataset
from src.calibration.mono.calibrator import MonoCalibrator
from src.calibration.mono.constants import FixedPointMode
from src.calibration.reporting.table import print_calibration_comparison


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
DATA_DIR = "data/calibration_images"
LEFT_CAMERA_DIR_NAME = "left_cam"
RIGHT_CAMERA_DIR_NAME = "right_cam"

IMAGE_SIZE = (1280, 720)

# Calibration flags
CALIBRATOR_FLAGS = None
# Optional examples:
# CALIBRATOR_FLAGS = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
# CALIBRATOR_FLAGS |= cv2.CALIB_USE_INTRINSIC_GUESS

# Termination criteria for calibration optimization
CALIBRATOR_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

left_dir_path = f"{DATA_DIR}/{LEFT_CAMERA_DIR_NAME}"
right_dir_path = f"{DATA_DIR}/{RIGHT_CAMERA_DIR_NAME}"


# -------------------------------
# Initialize Calibration Pattern
# -------------------------------
pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=REFINE_CORNERS, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=SUBPIX_CRITERIA, square_size=SQUARE_SIZE)


# -------------------------------
# Load Dataset and Extract Features
# -------------------------------
# Load left camera images and detect corners
data_set = DirectoryCalibrationDataset(data_dir=left_dir_path, pattern=pattern)
left_image_points  = data_set.extract_features()

# Load right camera images and detect corners
data_set.set_data_dir(data_dir=right_dir_path)
right_image_points = data_set.extract_features()


# -------------------------------
# Initialize Calibrator and Run Calibration
# -------------------------------
calibrator = MonoCalibrator(pattern=pattern, image_size=IMAGE_SIZE, image_points=left_image_points, i_fixed_point=FixedPointMode.TOP_RIGHT, flags=CALIBRATOR_FLAGS, criteria=CALIBRATOR_CRITERIA)

# Calibrate left camera
left_calibration_result = calibrator.calibrate()

# Calibrate right camera using its image points
right_calibration_result = calibrator.calibrate(image_points=right_image_points)


# -------------------------------
# Display Calibration Comparison
# -------------------------------
# Compare left and right camera intrinsic and extrinsic parameters
print_calibration_comparison(left=left_calibration_result, right=right_calibration_result)
