import numpy as np
from pathlib import Path

from .types import CalibrationNpzData
from .base import CalibrationStorage
from src.calibration.mono.types import CalibrationResult
from src.utils.path import ensure_file_path


class NpzCalibrationStorage(CalibrationStorage[CalibrationResult]):
    def save(self, result: CalibrationResult, filename: Path | str) -> None:
        file_path = ensure_file_path(file_path=filename)

        np.savez(
            file=file_path,
            rms=result.rms,
            camera_matrix=result.camera_matrix,
            dist_coeffs=result.dist_coeffs,
            rvecs=result.rvecs,
            tvecs=result.tvecs,
            object_points=result.object_points,
            std_intrinsics=result.std_intrinsics,
            std_extrinsics=result.std_extrinsics,
            std_object_points=result.std_object_points,
            per_view_error=result.per_view_error
        )


    def load(self, filename: Path | str) -> CalibrationResult:
        file_path = ensure_file_path(file_path=filename)

        data: CalibrationNpzData = np.load(file_path, allow_pickle=True)

        return CalibrationResult(
            rms=float(data['rms']),
            camera_matrix=data['camera_matrix'],
            dist_coeffs=data['dist_coeffs'],
            rvecs=list(data['rvecs']),
            tvecs=list(data['tvecs']),
            object_points=data['object_points'],
            std_intrinsics=data['std_intrinsics'],
            std_extrinsics=data['std_extrinsics'],
            std_object_points=data['std_object_points'],
            per_view_error=data['per_view_error']
        )
