from typing import TypedDict
import numpy as np
from numpy.typing import NDArray


class CalibrationNpzData(TypedDict):
    rms: NDArray[np.float64]
    camera_matrix: NDArray[np.float64]
    dist_coeffs: NDArray[np.float64]
    rvecs: NDArray[np.float64]
    tvecs: NDArray[np.float64]
    object_points: NDArray[np.float64]
    std_intrinsics: NDArray[np.float64]
    std_extrinsics: NDArray[np.float64]
    std_object_points: NDArray[np.float64]
    per_view_error: NDArray[np.float64]
