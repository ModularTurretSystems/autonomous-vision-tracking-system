"""
src/calibration/stereo/calibrator.py

Author
------
Stanislav : original implementation (2026-01-31)
Danil4615 : refactoring and documentation (2026-02-07)

Description
-----------
This module provides a class-based interface for stereo camera calibration
based on previously computed mono calibration results for the left and right
cameras.

The main entry point is the `StereoCalibrator` class, which wraps OpenCV's
`stereoCalibrateExtended` function and returns a structured
`StereoCalibrationResult` object.
"""


import cv2
import numpy as np

from src.calibration.patterns.chessboard import ChessboardPattern

from typing import Optional, Tuple
from cv2.typing import MatLike, Size, TermCriteria
from src.calibration.mono.types import MonoCalibrationResult
from .types import StereoCalibrationResult


class StereoCalibrator():
    """
    Perform stereo camera calibration using mono calibration results.

    This class takes calibration results for the left and right cameras,
    along with a known calibration pattern, and estimates the extrinsic
    relationship between the two cameras.

    The calibration process computes:
        - Refined intrinsic parameters
        - Rotation and translation between cameras
        - Essential and fundamental matrices
        - Per-view reprojection errors

    Notes
    -----
    This class assumes that both mono calibration results were obtained
    using the same calibration pattern and the same set of views.
    """

    def __init__(
        self,
        left_calibration_data: MonoCalibrationResult,
        right_calibration_data: MonoCalibrationResult,
        pattern: ChessboardPattern,
        image_size: Size,
        *,
        flags: Optional[int] = None,
        criteria: Optional[TermCriteria] = None
    ) -> None:
        """
        Initialize the stereo calibrator.

        Parameters
        ----------
        left_calibration_data : MonoCalibrationResult
            Mono calibration result for the left camera.
        right_calibration_data : MonoCalibrationResult
            Mono calibration result for the right camera.
        pattern : ChessboardPattern
            Calibration pattern used to generate object points.
        image_size : Size
            Size of the calibration images as (width, height).
        flags : int, optional
            OpenCV stereo calibration flags.
        criteria : TermCriteria, optional
            Termination criteria for the stereo calibration optimizer.
        """
        self.left_calibration_data = left_calibration_data
        self.right_calibration_data = right_calibration_data
        self.pattern = pattern
        self.image_size = image_size
        self.flags = flags
        self.criteria = criteria
        self.objps = pattern.generate_objps(len(left_calibration_data.image_points))

    
    def set_flags(self, flags: int) -> None:
        """
        Set OpenCV stereo calibration flags.

        Parameters
        ----------
        flags : int
            OpenCV stereo calibration flags (e.g. cv2.CALIB_FIX_INTRINSIC).
        """
        self.flags = flags


    def set_criteria(self, cirteria: TermCriteria) -> None:
        """
        Set termination criteria for stereo calibration.

        Parameters
        ----------
        criteria : TermCriteria
            OpenCV termination criteria tuple.
        """
        self.criteria = cirteria


    def _normalize_calibration_args(
        self
    ) -> Tuple[int, TermCriteria]:
        """
        Normalize optional calibration arguments.

        Returns
        -------
        tuple of (int, TermCriteria)
            Tuple containing flags and termination criteria suitable
            for passing into OpenCV's stereo calibration function.
        """
        return self.flags, self.criteria #type: ignore


    def calibrate(self) -> StereoCalibrationResult:
        """
        Run stereo camera calibration.

        This method wraps OpenCV's `stereoCalibrateExtended` and returns
        the result in a structured `StereoCalibrationResult` object.

        Returns
        -------
        StereoCalibrationResult
            Stereo calibration result containing:
                - Refined intrinsic parameters
                - Rotation matrix and translation vector
                - Essential and fundamental matrices
                - Per-view reprojection errors
                - Image points used for calibration
        """
        rotation_matrix: MatLike = np.array([])
        translation_vector: MatLike = np.array([])
        flags, criteria = self._normalize_calibration_args()

        rms, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, rotation_matrix, translation_vector, essential_matrix, fundamental_matrix, rvecs, tvecs, per_view_errors = cv2.stereoCalibrateExtended(
            objectPoints=self.objps,
            imagePoints1=self.left_calibration_data.image_points,
            imagePoints2=self.right_calibration_data.image_points,
            cameraMatrix1=self.left_calibration_data.camera_matrix,
            distCoeffs1=self.right_calibration_data.dist_coeffs,
            cameraMatrix2=self.right_calibration_data.camera_matrix,
            distCoeffs2=self.right_calibration_data.dist_coeffs,
            imageSize=self.image_size,
            R=rotation_matrix,
            T=translation_vector,
            flags=flags,
            criteria=criteria
        )

        return StereoCalibrationResult(
            rms=rms,
            left_camera_matrix=left_camera_matrix,
            left_dist_coeffs=left_dist_coeffs,
            right_camera_matrix=right_camera_matrix,
            right_dist_coeffs=right_dist_coeffs,
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
            essential_matrix=essential_matrix,
            fundamental_matrix=fundamental_matrix,
            rvecs=rvecs,
            tvecs=tvecs,
            per_view_errors=per_view_errors,
            left_image_points=self.left_calibration_data.image_points,
            right_image_points=self.right_calibration_data.image_points
        )
