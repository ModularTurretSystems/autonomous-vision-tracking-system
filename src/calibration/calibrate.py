import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from src.utils.camera_utils import estimate_AOV

from cv2.typing import MatLike
from src.calibration.types import CalibratedData


# ========== CONSTANTS for out() ==========
NPZ_FILE_PATH = "data/calibration_results/calibration_points.npz"
TEST_LEFT_PATH = "data/calibration_images/left_cam/"
TEST_RIGHT_PATH = "data/calibration_images/right_cam/"
IMAGE_SIZE = (640, 480)

# ========== CONSTANTS for load_points_data() ==========
LPD_PATTERN_SIZE_DV = (9, 6)
LPD_SQUARE_SIZE_MM = 30.0
LPD_COLLECTED_GOOD_DV = 0
LPD_IMAGE_SIZE_DV = (640, 480)

# ========== CONSTANTS for stereo_calibrate() ==========
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
FLAGS = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 |cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5

# ========== CONSTANTS for compute_rectification_maps()
ALPHA_VALUE = 0
M1TYPE = cv2.CV_32FC1

# ========== CONSTANTS for show_disparity_map() ==========
WINDOW_SCALE = 0.7

MIN_DISPARITY = 0
NUM_DISPARITIES = 64
BLOCK_SIZE = 5
P1 = 8 * 3 * 5**2
P2 = 32 * 3 * 5**2
DISP12_MAX_DIFF = 1
UNIQUENESS_RATIO = 10
SPECKLE_WINDOW_SIZE = 100
SPECKLE_RANGE = 32

UNKNOWN_DIVISON_COEFF = 16.0

ALPHA = 0
BETA = 255

POINTS_3D_EXAMPLE_AREA_H = 200
POINTS_3D_EXAMPLE_AREA_W = 300

# ==================================================


def load_points_data(npz_path: str | Path) -> Dict[str, Any]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"File wasn't found: {npz_path}")

    print(f"Download calibration points: {npz_path.resolve()}")
    data: Any = np.load(npz_path, allow_pickle=True)

    return {
        "obj_points": data["obj_points"],
        "img_points_left": data["img_points_left"],
        "img_points_right": data["img_points_right"],
        "pattern_size": tuple(data.get("pattern_size", LPD_PATTERN_SIZE_DV)),
        "square_size_mm": float(data.get("square_size_mm", LPD_SQUARE_SIZE_MM)),
        "collected_good": int(data.get("collected_good", LPD_COLLECTED_GOOD_DV)),
        "image_size": tuple(data.get("image_size", LPD_IMAGE_SIZE_DV)),
    }


def stereo_calibrate(
        data: Dict[str, Any]
    ) -> CalibratedData:
    image_size = data["image_size"]
    objpoints = data["obj_points"]
    imgpoints_l = data["img_points_left"]
    imgpoints_r = data["img_points_right"]

    criteria = CRITERIA

    flags = FLAGS

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

    AOV_params = estimate_AOV(matx=mtx_l, img_size=IMAGE_SIZE)

    return CalibratedData(
        rms=ret,
        matx_l=mtx_l,   
        dist_l=dist_l,
        matx_r=mtx_r,
        dist_r=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        AOV = AOV_params
    )


def compute_rectification_maps(
    calib: CalibratedData,
    image_size: Tuple[int, int]
) -> Tuple[Tuple[MatLike, MatLike], Tuple[MatLike, MatLike], MatLike]:
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        calib.matx_l, calib.dist_l,
        calib.matx_r, calib.dist_r,
        image_size, calib.R, calib.T, alpha=ALPHA_VALUE
    )

    map_l_x, map_l_y = cv2.initUndistortRectifyMap(
        cameraMatrix=calib.matx_l,
        distCoeffs=calib.dist_l,
        R=R1,
        newCameraMatrix=P1,
        size=image_size,
        m1type=M1TYPE
    )
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(
        cameraMatrix=calib.matx_r,
        distCoeffs=calib.dist_r,
        R=R2,
        newCameraMatrix=P2,
        size=image_size,
        m1type=M1TYPE
    )

    return (map_l_x, map_l_y), (map_r_x, map_r_y), Q


def show_disparity_map(
    img_l: MatLike,
    img_r: MatLike,
    map_l: Tuple[MatLike, MatLike],
    map_r: Tuple[MatLike, MatLike],
    Q: MatLike,
    img_num: int,
    window_scale: float = WINDOW_SCALE
    ):
    """Показывает rectified изображения и карту глубины"""
    # Rectify
    rect_l = cv2.remap(img_l, map_l[0], map_l[1], cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map_r[0], map_r[1], cv2.INTER_LINEAR)
    
    # Смотрим на rectified пару рядом
    combined_rect = np.hstack((rect_l, rect_r))
    cv2.imshow(f"Rectified (left | right) img {img_num}", cv2.resize(combined_rect, None, fx=window_scale, fy=window_scale))
    cv2.moveWindow(f"Rectified (left | right) img {img_num}", 550, 50)
    stereo = cv2.StereoSGBM.create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=P1,
        P2=P2,
        disp12MaxDiff=DISP12_MAX_DIFF,
        uniquenessRatio=UNIQUENESS_RATIO,
        speckleWindowSize=SPECKLE_WINDOW_SIZE,
        speckleRange=SPECKLE_RANGE
    )

    disparity = stereo.compute(rect_l, rect_r).astype(np.float32) / UNKNOWN_DIVISON_COEFF

    # Нормализация для отображения
    disp_vis = np.empty_like(disparity, dtype=np.uint8)
    cv2.normalize(disparity, disp_vis, alpha=ALPHA, beta=BETA, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    cv2.imshow(f"Disparity (grayscale) img {img_num}", cv2.resize(disp_vis, None, fx=window_scale, fy=window_scale))
    cv2.moveWindow(f"Disparity (grayscale) img {img_num}", 50, 400)
    cv2.imshow(f"Disparity (color map) img {img_num}", cv2.resize(disp_color, None, fx=window_scale, fy=window_scale))
    cv2.moveWindow(f"Disparity (color map) img {img_num}", 50, 50)
    # Опционально: 3D-точки (point cloud)
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    print("Пример 3D-точки (x,y,z) в мм:", points_3d[POINTS_3D_EXAMPLE_AREA_H, POINTS_3D_EXAMPLE_AREA_W])  # пример точки

    cv2.waitKey(500)
    cv2.destroyAllWindows()


def out():
    npz_file = NPZ_FILE_PATH

    test_left_path = TEST_LEFT_PATH
    test_right_path = TEST_RIGHT_PATH

    try:
        points_data = load_points_data(npz_file)

        calib = stereo_calibrate(points_data)

        print(f"RMS: {calib.rms:.4f}")

        maps_l, maps_r, Q = compute_rectification_maps(calib, image_size=IMAGE_SIZE)

        count_photos = len([f for f in Path(test_left_path).iterdir() if f.suffix.lower() == '.jpg'])

        for num in range(count_photos):
            test_l = cv2.imread(str(Path(test_left_path) / f"{num}.jpg"))
            test_r = cv2.imread(str(Path(test_right_path) / f"{num}.jpg"))

            if test_l is None or test_r is None: #type: ignore
                print("Не удалось загрузить тестовые кадры — пропускаем disparity map")
            else:
                print("\nПоказываем карту глубины...")
                show_disparity_map(test_l, test_r, maps_l, maps_r, Q, num)
           
            
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    out()
