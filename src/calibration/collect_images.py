# calibration/collect_images.py

from src.vision.stereo.calibration_session import StereoCalibrationSession
from src.vision.stereo.capture import StereoCapture
from src.vision.stereo.system import StereoSystem
import os
import numpy as np
from pathlib import Path


def collect_images() -> None:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

    stereo_system = StereoSystem(left=0, right=1)
    save_dir = Path("data/calibration_images")
    save_dir.mkdir(exist_ok=True, parents=True)

    capture = StereoCapture(stereo_system=stereo_system, save_dir=save_dir)

    session = StereoCalibrationSession(
        stereo_capture=capture,
        pattern_size=(9, 6),
        min_good_frames=15,
        max_good_frames=40
    )

    print("\n=== Запуск сессии калибровки ===")
    print("Управление:")
    print(" s → сохранить хороший кадр (если доска найдена)")
    print(" q → выйти")

    result = session.run()

    print("\n" + "="*40)
    print("Сессия завершена")
    print(f"Собрано хороших кадров: {result.get('collected_good', 0)}")
    print(f"Сохранено файлов: {result.get('saved_images_count', 0)}")
    print(f"Готово к калибровке: {result.get('enough', False)}")
    print("="*40)

    if result.get('enough', False):
        output_dir = Path("data/calibration_results")
        output_dir.mkdir(exist_ok=True, parents=True)

        output_file = output_dir / f"calibration_points.npz"

        np.savez(
            output_file,
            obj_points=result["obj_points"],
            img_points_left=result["img_points_left"],
            img_points_right=result["img_points_right"],
            collected_good=result.get("collected_good", 0),
            pattern_size=(9, 6),
            square_size_mm=30.0,
            
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