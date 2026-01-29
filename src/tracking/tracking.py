from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from src.camera.camera import Camera
from src.tracking.yolo_tracker import YoloTargetDetector
from src.utils.types import CFG

# =========================
# CONFIG
# =========================
CALIBRATION_OUTPUT_PATH = "data/calibration_results/final_stereo_calibration.npz"

CAM_L_ID = CFG.CAM_L_ID
CAM_R_ID = CFG.CAM_R_ID

FRAME_W, FRAME_H = CFG.FRAME_SIZE

EXIT_BUTTON = "q"



# =========================
# IO: calibration load
# =========================
def load_final_calibration_data(npz_path: str | Path) -> Dict[str, Any]:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Файл калибровки не найден: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    out: Dict[str, Any] = {
        "map_l": (data["map_l_x"], data["map_l_y"]),
    }

    # ниже нужно только для стерео
    if "map_r_x" in data.files and "map_r_y" in data.files:
        out["map_r"] = (data["map_r_x"], data["map_r_y"])
    if "Q" in data.files:
        out["Q"] = data["Q"]

    return out


def tracking() -> None:
    det = YoloTargetDetector(
        model_path="yolo26n.pt",
        conf_th=0.5,
        classes=[0],                 # person
        pick_mode="closest_center",  # "largest" / "highest_conf" / "closest_center"
        aim_point="bbox_center",           # "bbox_center" / "head" / "torso"
        ema_alpha=0.35,
    )
    
    camera_l = Camera(
        camera_id=CAM_L_ID,
        apiPreference=None,
        params=(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H, cv2.CAP_PROP_FRAME_WIDTH, FRAME_W),
    )

    calib = load_final_calibration_data(CALIBRATION_OUTPUT_PATH)
    maps_l = calib["map_l"]

    # maps_r = calib.get("map_r", None)
    # Q = calib.get("Q", None)


    print(f"Трекинг запущен. Нажмите '{EXIT_BUTTON}' для выхода.")


    while True:
        frame = camera_l.capture_frame()

        frame_l = frame.frame

        rect_l = cv2.remap(frame_l, maps_l[0], maps_l[1], cv2.INTER_LINEAR)

        pt, bbox, dbg = det.get_target(
            frame_bgr=rect_l,
            return_debug=False
        )

        dz = det.deadzone_rect(
            frame_bgr=rect_l,
            deadzone_frac=(0.15, 0.15)
        )

        inside = (pt is not None) and det.point_in_dz(pt_xy=pt, dz_xyxy=dz)
        vis = det.draw(
            frame_bgr=rect_l,
            det=bbox,
            pt=pt, 
            dz=dz
        )

        cv2.putText(
            img=vis,
            text=f"inside_deadzone: {inside}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.8,
            color=(255, 255, 0),
            thickness=2
        )  

        
        cv2.imshow("res", vis)


           
        k = cv2.waitKey(1) & 0xFF
        if k == ord(EXIT_BUTTON):
            break



if __name__ == "__main__":
    tracking()
