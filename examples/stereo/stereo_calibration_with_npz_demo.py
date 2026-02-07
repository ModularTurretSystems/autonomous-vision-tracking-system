"""
examples/stereo/stereo_calibration_with_npz_demo.py

Author: Danil4615
Date: 2026-02-07

Description
-----------
This script demonstrates how to perform stereo camera calibration using previously
computed mono calibration results stored in NPZ files. It uses the `StereoCalibrator`
class to compute the extrinsic parameters between two cameras, as well as essential
and fundamental matrices, RMS reprojection error, and per-view errors.

The resulting stereo calibration is then saved to an NPZ file for future use.

Usage
-----
python -m examples.stereo.stereo_calibration_with_npz_demo
"""


import cv2

from src.calibration.persistence.npz import NpzCalibrationStorage
from src.calibration.stereo.types import StereoCalibrationResult
from src.calibration.mono.types import MonoCalibrationResult
from src.calibration.stereo.calibrator import StereoCalibrator
from src.calibration.patterns.chessboard import ChessboardPattern


# ===============================
# USER CONFIGURATION
# ===============================
MAIN_DIR = "data/npz_stereo"
LEFT_FILENAME = "left.npz"
RIGHT_FILENAME = "right.npz"

# File path to save the resulting stereo calibration
STEREO_SAVE_PATH  = "data/npz/stereo.npz"

# Stereo calibration pattern settings
PATTERN_SIZE = (9, 6)   # Number of inner corners per chessboard row and column
SQUARE_SIZE = 30.0      # Size of a square in mm

# Image size used during calibration
IMAGE_SIZE = (1280, 720)

# OpenCV stereo calibration flags
FLAGS = None
# Example of optional flags:
# FLAGS = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
# FLAGS |= cv2.CALIB_USE_INTRINSIC_GUESS

# Termination criteria for stereo calibration optimization
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)


# ===============================
# Load Mono Calibration Results
# ===============================
mono_storage = NpzCalibrationStorage(MonoCalibrationResult)
left_path  = f"{MAIN_DIR}/{LEFT_FILENAME}"
right_path  = f"{MAIN_DIR}/{RIGHT_FILENAME}"

left_data = mono_storage.load(filename=left_path)
right_data = mono_storage.load(filename=right_path)

# ===============================
# Create Chessboard Pattern Object
# Used to generate 3D object points for stereo calibration
# ===============================
pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, square_size=SQUARE_SIZE)
pattern = ChessboardPattern(pattern_size=PATTERN_SIZE,square_size=SQUARE_SIZE)

# ===============================
# Initialize Stereo Calibrator
# ===============================
stereo_calibrator = StereoCalibrator(
    left_calibration_data=left_data, 
    right_calibration_data=right_data, 
    pattern=pattern, 
    image_size=IMAGE_SIZE, 
    flags=FLAGS, 
    criteria=CRITERIA
)

# ===============================
# Run Stereo Calibration
# ===============================
stereo_calibration_result = stereo_calibrator.calibrate()

# ===============================
# Display Calibration Results
# ===============================
print(stereo_calibration_result.rms)
print('='*50)
print(stereo_calibration_result.per_view_errors)

# ===============================
# Save Stereo Calibration Results
# ===============================
stereo_storage = NpzCalibrationStorage(StereoCalibrationResult)
stereo_storage.save(result=stereo_calibration_result, filename=STEREO_SAVE_PATH )
print(f"[INFO] Stereo calibration saved to {STEREO_SAVE_PATH }")
