from dataclasses import dataclass

from cv2.typing import MatLike
from typing import Sequence


@dataclass
class StereoCalibrationResult:
    rms: float
    left_camera_matrix: MatLike
    left_dist_coeffs: MatLike
    right_camera_matrix: MatLike
    right_dist_coeffs: MatLike
    rotation_matrix: MatLike
    translation_vector: MatLike
    essential_matrix: MatLike
    fundamental_matrix: MatLike
    rvecs: Sequence[MatLike]
    tvecs: Sequence[MatLike]
    per_view_errors: MatLike
    left_image_points: Sequence[MatLike]
    right_image_points: Sequence[MatLike]
