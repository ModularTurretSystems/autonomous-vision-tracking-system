import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from src.calibration.types import CalibratedData


def load_points_data(npz_path: str | Path) -> Dict[str, Any]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"File wasn't found: {npz_path}")

    print(f"Download calibration points: {npz_path.resolve()}")
    data = np.load(npz_path, allow_pickle=True)

    return {
        "obj_points": data["obj_points"],
        "img_points_left": data["img_points_left"],
        "img_points_right": data["img_points_right"],
        "pattern_size": tuple(data.get("pattern_size", (9, 6))),
        "square_size_mm": float(data.get("square_size_mm", 30.0)),
        "collected_good": int(data.get("collected_good", 0)),
        "image_size": tuple(data.get("image_size", (1280, 720))),
    }


def stereo_calibrate(
        data: Dict[str, Any]
    ) -> CalibratedData:
    image_size = data["image_size"]
    objpoints = data["obj_points"]
    imgpoints_l = data["img_points_left"]
    imgpoints_r = data["img_points_right"]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 |cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

    # Create empty matrix
    empty_camera = np.eye(3, dtype=np.float64)
    empty_dist = np.zeros(5, dtype=np.float64)

    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpoints_l,
        imagePoints2=imgpoints_r,
        cameraMatrix1=empty_camera,           
        distCoeffs1=empty_dist,               
        cameraMatrix2=empty_camera,
        distCoeffs2=empty_dist,
        imageSize=image_size,
        flags=flags,
        criteria=criteria
    )

    return CalibratedData(
        rms=ret,
        matx_l=mtx_l,
        dist_l=dist_l,
        matx_r=mtx_r,
        dist_r=dist_r,
        R=R,
        T=T,
        E=E,
        F=F
    )

def compute_rectification_maps(
    calib: CalibratedData,
    image_size: Tuple[int, int]
) -> Tuple:
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib.matx_l, calib.dist_l,
        calib.matx_r, calib.dist_r,
        image_size, calib.R, calib.T, alpha=0
    )

    map_l_x, map_l_y = cv2.initUndistortRectifyMap(
        calib.matx_l, calib.dist_l, R1, P1, image_size, cv2.CV_32FC1
    )
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(
        calib.matx_r, calib.dist_r, R2, P2, image_size, cv2.CV_32FC1
    )

    return (map_l_x, map_l_y), (map_r_x, map_r_y), Q


def show_disparity_map(
    img_l: np.ndarray,
    img_r: np.ndarray,
    map_l: Tuple[np.ndarray, np.ndarray],
    map_r: Tuple[np.ndarray, np.ndarray],
    Q: np.ndarray,
    window_scale: float = 0.7
):
    """Показывает rectified изображения и карту глубины"""
    # Rectify
    rect_l = cv2.remap(img_l, map_l[0], map_l[1], cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map_r[0], map_r[1], cv2.INTER_LINEAR)

    # Смотрим на rectified пару рядом
    combined_rect = np.hstack((rect_l, rect_r))
    cv2.imshow("Rectified (left | right)", cv2.resize(combined_rect, None, fx=window_scale, fy=window_scale))

    stereo = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(rect_l, rect_r).astype(np.float32) / 16.0

    # Нормализация для отображения
    disp_vis = np.empty_like(disparity, dtype=np.uint8)
    cv2.normalize(disparity, disp_vis, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    cv2.imshow("Disparity (grayscale)", cv2.resize(disp_vis, None, fx=window_scale, fy=window_scale))
    cv2.imshow("Disparity (color map)", cv2.resize(disp_color, None, fx=window_scale, fy=window_scale))

    # Опционально: 3D-точки (point cloud)
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    print("Пример 3D-точки (x,y,z) в мм:", points_3d[200, 300])  # пример точки

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def out():
    # Укажи свой файл (из collect_images.py)
    npz_file = "data/calibration_results/calibration_points.npz"  # ← измени на актуальный

    test_left_path = "data/calibration_images/left_cam/1.jpg"
    test_right_path = "data/calibration_images/right_cam/1.jpg"

    try:
        points_data = load_points_data(npz_file)

        calib = stereo_calibrate(points_data)

        print(f"RMS: {calib.rms:.4f}")

        maps_l, maps_r, Q = compute_rectification_maps(calib, image_size=(1280, 720))

        test_l = cv2.imread(test_left_path)
        test_r = cv2.imread(test_right_path)

        if test_l is None or test_r is None:
            print("Не удалось загрузить тестовые кадры — пропускаем disparity map")
        else:
            print("\nПоказываем карту глубины...")
            show_disparity_map(test_l, test_r, maps_l, maps_r, Q)
            
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    out()