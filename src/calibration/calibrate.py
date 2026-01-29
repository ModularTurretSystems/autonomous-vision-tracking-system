import cv2
import numpy as np

from src.utils.types import CFG
from pathlib import Path
from typing import Dict, Any, Tuple
from cv2.typing import MatLike
from .types import CalibratedData

# ========== CONSTANTS for out() ==========
NPZ_FILE_PATH = "data/calibration_results/calibration_points.npz"
CALIBRATION_OUTPUT_PATH = "data/calibration_results/final_stereo_calibration.npz"
IMAGE_SIZE = CFG.FRAME_SIZE

# ========== CONSTANTS for load_points_data() ==========
LPD_PATTERN_SIZE_DV = (9, 6)
LPD_SQUARE_SIZE_MM = 30.0
LPD_COLLECTED_GOOD_DV = 0
LPD_IMAGE_SIZE_DV = CFG.FRAME_SIZE

# ========== CONSTANTS for stereo_calibrate() ==========
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
FLAGS = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 
IFIXEDPOINT = (LPD_PATTERN_SIZE_DV[0]//2) * LPD_PATTERN_SIZE_DV[0] + (LPD_PATTERN_SIZE_DV[1]//2)

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

    _, mtx_l, dist_l, _, _, _ = cv2.calibrateCameraRO(
        objectPoints=objpoints,
        imagePoints=imgpoints_l,
        imageSize= IMAGE_SIZE,
        iFixedPoint= IFIXEDPOINT,
        cameraMatrix=empty_camera,
        distCoeffs= empty_dist,
        flags=FLAGS,
        criteria=CRITERIA
    )

    _, mtx_r, dist_r, _, _, _ = cv2.calibrateCameraRO(
        objectPoints=objpoints,
        imagePoints=imgpoints_r,
        imageSize= IMAGE_SIZE,
        iFixedPoint= IFIXEDPOINT,
        cameraMatrix=empty_camera,
        distCoeffs= empty_dist,
        flags=FLAGS,
        criteria=CRITERIA
    )

    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpoints_l,
        imagePoints2=imgpoints_r,
        cameraMatrix1=mtx_l,           
        distCoeffs1=dist_l,               
        cameraMatrix2=mtx_r,
        distCoeffs2=dist_r,
        imageSize=image_size,
        flags=flags,
        criteria=criteria
    )

    # AOV_params = estimate_AOV(matx=mtx_l, img_size=IMAGE_SIZE)

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


def save_calibration(
) -> None:
    
    output_path = Path(CALIBRATION_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npz_file = NPZ_FILE_PATH

    try:
        points_data = load_points_data(npz_path=npz_file)
        
        calib = stereo_calibrate(data=points_data)

        maps_l, maps_r, Q = compute_rectification_maps(calib=calib, image_size=IMAGE_SIZE)
    
        print(f"RMS: {calib.rms:.4f}")

        np.savez(
        output_path,
        map_l_x=maps_l[0],  
        map_l_y=maps_l[1],
        map_r_x=maps_r[0],
        map_r_y=maps_r[1],
        Q=Q
    )

    except Exception as e:
        print(f"❌ Ошибка при сохранении калибровки: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    save_calibration()
