# src/calibration/collect_images.py

import cv2
import os
import numpy as np
from pathlib import Path

from src.vision.stereo.system import StereoSystem
from src.vision.stereo.capture import StereoCapture
from src.calibration.patterns.chessboard import ChessboardPattern
from src.vision.stereo.calibration_session import StereoCalibrationSession


# ========== CONSTANTS for collect_images() ==========
CAM_L_ID = 0
CAM_R_ID = 1

PATH = "data/calibration_images"

PATTERN_SIZE = (9, 6)
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
REFINE = True
WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

RESOLUTION = (640, 480)
MIN_GOOD_FRAMES = 15
MAX_GOOD_FRAMES = 40

SQUARE_SIZE_MM = 30.0

# ==================================================


def collect_images() -> None:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

    stereo_system = StereoSystem(left=CAM_L_ID, right=CAM_R_ID)
    save_dir = Path(PATH)
    save_dir.mkdir(exist_ok=True, parents=True)

    capture = StereoCapture(stereo_system=stereo_system, save_dir=save_dir)

    pattern = ChessboardPattern(
        pattern_size=PATTERN_SIZE,
        flags=FLAGS,
        refine=REFINE,
        win_size=WIN_SIZE,
        zero_zone=ZERO_ZONE,
        criteria=CRITERIA
    )

    session = StereoCalibrationSession(
        stereo_capture=capture,
        pattern=pattern,
        combined_resolution=RESOLUTION,
        min_good_frames=MIN_GOOD_FRAMES,
        max_good_frames=MAX_GOOD_FRAMES
    )

    print("\n=== Запуск сессии калибровки ===")
    print("Управление:")
    print(" s → сохранить хороший кадр (если доска найдена)")
    print(" q → выйти")

    result = session.run()

    print("\n" + "="*40)
    print("Сессия завершена")
    print(f"Собрано хороших кадров: {result.collected}")
    print(f"Сохранено файлов: {result.saved_images_count}")
    print(f"Готово к калибровке: {result.enough}")
    print("="*40)

    if result.enough:
        output_dir = Path("data/calibration_results")
        output_dir.mkdir(exist_ok=True, parents=True)

        output_file = output_dir / f"calibration_points.npz"

        np.savez(
            output_file,
            obj_points=result.obj_points,
            img_points_left=result.img_points_left,
            img_points_right=result.img_points_right,
            collected_good=result.collected,
            pattern_size=PATTERN_SIZE,
            square_size_mm=SQUARE_SIZE_MM,
        )

        print(f"\nТочки калибровки сохранены в:")
        print(f"  {output_file.resolve()}")
        print("Запустите калибровку:")
        print(f"  python calibration/calibrate.py")
    else:
        print("Недостаточно хороших кадров — продолжите сбор")


if __name__ == "__main__":
    try:
        collect_images()
    except KeyboardInterrupt:
        print("\nПрервано пользователем (Ctrl+C)")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
