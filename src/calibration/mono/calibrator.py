
from cv2 import initCameraMatrix2D, calibrateCameraROExtended
from src.calibration.mono.types import CalibrationResult
from src.calibration.patterns.chessboard import ChessboardPattern

from .constants import FixedPointMode

from cv2.typing import MatLike, Size, TermCriteria
from typing import Sequence


class MonoCalibrator():
    def __init__(
        self,
        pattern: ChessboardPattern,
        image_size: Size,
        image_points: Sequence[MatLike] | None = None,
        i_fixed_point: int | FixedPointMode = FixedPointMode.DEFAULT,
        *,
        camera_matrix: MatLike | None = None,
        dist_coeffs: MatLike | None = None,
        aspect_ratio: float = 1.0,
        flags: int | None = None,
        criteria: TermCriteria | None = None
    ) -> None:
        self.pattern = pattern
        self.image_size = image_size
        self.objp = pattern.generate_objp()
        self.image_points = image_points
        self.i_fixed_point = self.get_fixed_point_index(mode=i_fixed_point)
        self.dist_coeffs = dist_coeffs
        self.camera_matrix = camera_matrix
        self.aspect_ratio = aspect_ratio
        self.flags = flags
        self.criteria = criteria

        self.objps: Sequence[MatLike] | None = None

        if image_points is not None:
            self.objps = self.generate_objps(image_points=image_points)

            self.camera_matrix = initCameraMatrix2D(
                objectPoints=self.objps,
                imagePoints=image_points,
                imageSize=image_size,
                aspectRatio=aspect_ratio
            )

    
    def generate_objps(
        self, 
        length: int | None = None, 
        image_points: Sequence[MatLike] | None = None, 
    ) -> Sequence[MatLike]: 
        if length is None:
            if image_points is None:
                if self.image_points is None:
                    raise ValueError(
                        "Error"
                    )
                image_points = self.image_points
            length = len(image_points)

        self.objps = self.pattern.generate_objps(length=length)

        return self.objps


    def get_fixed_point_index(
        self,
        mode: int | FixedPointMode
    ) -> int:
        cols, rows = self.pattern.pattern_size

        if mode == FixedPointMode.NONE: return -1
        elif mode == FixedPointMode.TOP_LEFT: return 1
        elif mode == FixedPointMode.TOP_RIGHT: return cols
        elif mode == FixedPointMode.BOTTOM_LEFT: return (rows - 1)* cols - 2
        elif mode == FixedPointMode.BOTTOM_RIGHT: return rows * cols - 2
        elif mode == FixedPointMode.MIDDLE: return (rows // 2) * cols + (cols // 2)
        elif mode < -1: raise ValueError(f"Unsupported FixedPointMode: {mode}")
        else: return mode


    def _normalize_calibration_args(
        self,
        image_points: Sequence[MatLike] | None = None,
        i_fixed_point: int | FixedPointMode | None = None,
        flags: int | None = None,
        criteria: TermCriteria | None = None
    ) -> tuple[Sequence[MatLike], int, MatLike, MatLike, int, TermCriteria]:
        if image_points is None:
            if self.image_points is None:
                raise ValueError(
                    "Calibration requires image points, but none were provided. "
                    "Pass 'image_points' to calibrate() or set them when creating the calibrator."
                )
            image_points = self.image_points

        if i_fixed_point is None: i_fixed_point = self.i_fixed_point
        
        if flags is None: flags = self.flags

        if criteria is None: criteria = self.criteria

        return image_points, i_fixed_point, self.camera_matrix, self.dist_coeffs, flags, criteria #type: ignore


    def calibrate(
        self,
        *,
        image_points: Sequence[MatLike] | None = None,
        i_fixed_point: int | FixedPointMode | None = None,
        flags: int | None = None,
        criteria: TermCriteria | None = None
    ) -> CalibrationResult:
        
        image_points, i_fixed_point, cameraMatrix, distCoeffs, flags, criteria = self._normalize_calibration_args(image_points=image_points, i_fixed_point=i_fixed_point, flags=flags, criteria=criteria)
        if self.objps is None: self.objps = self.generate_objps(image_points=image_points)
        
        rms, cameraMatrix, distCoeffs, rvecs, tvecs, objp, stdInt, stdExt, stdObj, perViewError = calibrateCameraROExtended(
            objectPoints=self.objps, 
            imagePoints=image_points, 
            imageSize=self.image_size, 
            iFixedPoint=i_fixed_point,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
            flags=flags,
            criteria=criteria
        )

        return CalibrationResult(
            rms=rms, 
            camera_matrix=cameraMatrix, 
            dist_coeffs=distCoeffs, 
            rvecs=rvecs, 
            tvecs=tvecs, 
            object_points=objp, 
            std_intrinsics=stdInt, 
            std_extrinsics=stdExt, 
            std_object_points=stdObj, 
            per_view_error=perViewError,
            image_points=image_points
        )
    