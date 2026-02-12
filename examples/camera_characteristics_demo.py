
"""
examples/camera_characteristics_demo.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
Demonstrates the computation of camera intrinsic parameters and
derived characteristics (field-of-view, focal length, principal point,
aspect ratio) using a pre-calibrated camera matrix.

Instructions
------------
To run this example from the project root directory, execute:
    python -m examples.camera_characteristics_demo
"""


from src.calibration.persistence.npz import NpzCalibrationStorage
from src.calibration.stereo.types import StereoCalibrationResult

from src.utils.camera_utils import compute_camera_parameters


# ===============================
# Constants / Configuration
# ===============================
DATA_PATH = "data/npz/stereo.npz"
IMAGE_SIZE = (1280, 720)


# ===============================
# Load Calibration Data
# ===============================
data = NpzCalibrationStorage(StereoCalibrationResult).load(filename=DATA_PATH)


# ===============================
# Compute Camera Characteristics
# ===============================
camera_params = compute_camera_parameters(camera_matrix=data.left_camera_matrix, image_size=IMAGE_SIZE)

# ===============================
# Display Results
# ===============================
print("Camera Characteristics Demo:")
print(f"Horizontal FOV: {camera_params.fovx:.2f} degrees")
print(f"Vertical FOV: {camera_params.fovy:.2f} degrees")
print(f"Focal Length: {camera_params.focal_length:.2f} pixels")
print(f"Principal Point: ({camera_params.principal_point[0]:.2f}, {camera_params.principal_point[1]:.2f}) pixels")
print(f"Aspect Ratio: {camera_params.aspect_ratio:.2f}")
