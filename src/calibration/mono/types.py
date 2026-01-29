from dataclasses import dataclass
from cv2.typing import MatLike
from typing import Sequence


@dataclass
class CalibrationResult:
    rms: float
    camera_matrix: MatLike
    dist_coeffs: MatLike
    rvecs: Sequence[MatLike]
    tvecs: Sequence[MatLike]
    object_points: MatLike
    std_intrinsics: MatLike
    std_extrinsics: MatLike
    std_object_points: MatLike
    per_view_error: MatLike
