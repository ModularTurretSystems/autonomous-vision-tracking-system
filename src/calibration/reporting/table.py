from src.utils.print_options import np_printoptions

from .constants import NAMES_OF_DIST_COEFFS

from src.calibration.mono.types import MonoCalibrationResult


def print_distortion_comparison(left: MonoCalibrationResult, right: MonoCalibrationResult) -> None:
    left_dc = left.dist_coeffs.flatten()
    right_dc = right.dist_coeffs.flatten()

    names = NAMES_OF_DIST_COEFFS

    for i in range(min(len(left_dc), len(right_dc))):
        name = names[i] if i < len(names) else f"c{i}"
        print(f"{f'Dist {name}':<25} | {left_dc[i]:<35.6f} | {right_dc[i]:<35.6f}")


@np_printoptions()
def print_calibration_comparison(left: MonoCalibrationResult, right: MonoCalibrationResult) -> None:
    print("=" * 100)
    print(f"{'PARAMETER':<25} | {'LEFT CAMERA':<35} | {'RIGHT CAMERA':<35}")
    print("=" * 100)

    print(f"{'RMS error':<25} | {left.rms:<35.6f} | {right.rms:<35.6f}")

    print(f"{'fx':<25} | {left.camera_matrix[0,0]:<35.4f} | {right.camera_matrix[0,0]:<35.4f}")
    print(f"{'fy':<25} | {left.camera_matrix[1,1]:<35.4f} | {right.camera_matrix[1,1]:<35.4f}")
    print(f"{'cx':<25} | {left.camera_matrix[0,2]:<35.4f} | {right.camera_matrix[0,2]:<35.4f}")
    print(f"{'cy':<25} | {left.camera_matrix[1,2]:<35.4f} | {right.camera_matrix[1,2]:<35.4f}")

    print_distortion_comparison(left=left, right=right)

    print("=" * 100)

    print("\nPer-view reprojection error:")
    print(f"{'View':<10} | {'Left':<15} | {'Right':<15}")
    print("-" * 45)
    for i, (l_err, r_err) in enumerate(zip(left.per_view_error, right.per_view_error)):
        print(f"{i+1:<10} | {float(l_err[0]):<15.6f} | {float(r_err[0]):<15.6f}")
