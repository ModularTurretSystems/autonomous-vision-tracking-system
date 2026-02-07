"""
examples/stereo/mono_calibration_with_npz_demo.py

Author: KrutayaBabka
Created: 2026-01-29
Last Modified: 2026-02-05

Description:
    Demonstrates a complete workflow for monocular camera calibration using a chessboard pattern
    and saving the results in .npz format. This example covers:
        1. Defining a calibration pattern with subpixel refinement.
        2. Loading images from a directory for left and right cameras.
        3. Extracting chessboard corners from images.
        4. Running camera calibration to compute intrinsic parameters, distortion coefficients,
           and per-view reprojection errors.
        5. Saving calibration results to .npz files.

Usage:
    python -m examples.stereo.mono_calibration_with_npz_demo
"""


import cv2

from src.calibration.mono.types import MonoCalibrationResult
from src.calibration.patterns.chessboard import ChessboardPattern
from src.calibration.dataset.directory import DirectoryCalibrationDataset
from src.calibration.mono.calibrator import MonoCalibrator
from src.calibration.mono.constants import FixedPointMode
from src.calibration.persistence.npz import NpzCalibrationStorage


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

# File path to save calibration results
MAIN_DIR = "data/npz_stereo"
LEFT_FILENAME = "left.npz"
RIGHT_FILENAME = "right.npz"


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
# Save Calibration Results to NPZ
# -------------------------------
# Build full paths for saving
left_path = f"{MAIN_DIR}/{LEFT_FILENAME}"
right_path = f"{MAIN_DIR}/{RIGHT_FILENAME}"

# Create storage instance for MonoCalibrationResult
mono_storage = NpzCalibrationStorage(MonoCalibrationResult)

# Save left and right calibration results
mono_storage.save(result=left_calibration_result, filename=left_path)
mono_storage.save(result=right_calibration_result, filename=right_path)
