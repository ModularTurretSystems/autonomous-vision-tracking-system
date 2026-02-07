
from src.calibration.mono.types import MonoCalibrationResult
from src.calibration.patterns.chessboard import ChessboardPattern

from .constants import FixedPointMode

from cv2.typing import MatLike, Size, TermCriteria
from typing import overload, Sequence


class MonoCalibrator():
    @overload
    def __init__(self,
        pattern: ChessboardPattern,
        image_size: Size,
        image_points: Sequence[MatLike],
        i_fixed_point: int | FixedPointMode = FixedPointMode.DEFAULT,
        *,
        camera_matrix: MatLike | None = None,
        dist_coeffs: MatLike | None = None,
        flags: int | None = None,
        criteria: TermCriteria | None = None,
    ) -> None: ...


    @overload
    def __init__(
        self,
        pattern: ChessboardPattern,
        image_size: Size,
        *,
        camera_matrix: MatLike | None = None,
        dist_coeffs: MatLike | None = None,
        flags: int | None = None,
        criteria: TermCriteria | None = None,
    ) -> None: ...

    
    @overload
    def __init__(
        self,
        pattern: ChessboardPattern,
        image_size: Size,
        image_points: Sequence[MatLike],
        *,
        aspect_ratio: float = 1.0,
        camera_matrix: MatLike | None = None,
        dist_coeffs: MatLike | None = None,
        flags: int | None = None,
        criteria: TermCriteria | None = None,
    ) -> None: ...


    @overload
    def generate_objps(self) -> Sequence[MatLike]: ...


    @overload
    def generate_objps(self, length: int) -> Sequence[MatLike]: ...


    @overload
    def generate_objps(self, *, image_points: Sequence[MatLike]) -> Sequence[MatLike]: ...


    @overload
    def calibrate(self) -> MonoCalibrationResult: ...


    @overload
    def calibrate(
        self, 
        *, 
        image_points: Sequence[MatLike] | None = None, 
        i_fixed_point: int | FixedPointMode | None = None, 
        flags: int | None = None, 
        criteria: TermCriteria | None = None
    ) -> MonoCalibrationResult: ...
    