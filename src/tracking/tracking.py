from pathlib import Path
from typing import Optional, Literal

import cv2
import numpy as np
import time

from src.actuation.arduino import ArduinoSerial
from src.actuation.turret import TurretController
from src.tracking import draw_debug
from src.tracking.errors import compute_aim_error
from src.tracking.types import VisionData
from src.tracking.yolo_tracker import YoloTargetDetector
from src.utils.types import CFG
from src.vision.stereo.system import StereoSystem
from .draw_debug import draw_debug, draw_timer

# =========================
# CONFIG
# =========================
CALIBRATION_OUTPUT_PATH = "data/calibration_results/final_stereo_calibration.npz"

CAM_L_ID = CFG.CAM_L_ID
CAM_R_ID = CFG.CAM_R_ID

EXIT_BUTTON = "q"

DEPTH_EVERY_N = 3  # считать глубину раз в N кадров (для FPS)
EMA_ALPHA = 0.35  # сглаживание глубины
MAX_Z_STEP = 100.0  # мм: максимум изменения глубины за один апдейт
LOST_RESET_AFTER = 1

Z_MIN = 500.0  # мм
Z_MAX = 6000.0  # мм

DEPTH_METHOD: Literal["sad", "sgbm"] = "sad"  # "sad" быстрее, "sgbm" точнее

SERIAL_PORT = "/dev/cu.usbserial-110"
SERIAL_BAUD = 115200

PAN_MIN, PAN_MAX = 20, 160
TILT_MIN, TILT_MAX = 30, 150

K_PAN = 1.2
K_TILT = 1.0

SERVO_SEND_PERIOD = 0.05  # 50 ms


def load_final_calibration_data(npz_path: str | Path) -> VisionData:
    data = np.load(npz_path, allow_pickle=True)
    return VisionData(
        map_l_x=data["map_l_x"].astype(np.float32),
        map_l_y=data["map_l_y"].astype(np.float32),
        map_r_x=data["map_r_x"].astype(np.float32),
        map_r_y=data["map_r_y"].astype(np.float32),
        Q=data["Q"].astype(np.float32),
        P1=data["P1"].astype(np.float32)
    )


def update_depth_filtered(
    z_prev: Optional[float],
    z_raw: Optional[float],
    max_step: float,
    ema_alpha: float,
) -> Optional[float]:
    """
    Если есть новое измерение z_raw:
      - если z_prev нет -> принять
      - если скачок большой -> двигаться ступенькой max_step
      - иначе -> EMA
    """
    if z_raw is None:
        return z_prev

    if z_prev is None:
        return z_raw

    delta = z_raw - z_prev
    if abs(delta) > max_step:
        return z_prev + float(np.sign(delta)) * max_step

    return ema_alpha * z_raw + (1.0 - ema_alpha) * z_prev


def tracking() -> None:
    det = YoloTargetDetector(
        model_path="yolo26n.pt",
        conf_th=0.5,
        classes=[0],
        pick_mode="closest_center",
        aim_point="head",
        ema_alpha=EMA_ALPHA,
    )

    arduino = ArduinoSerial(
        port=SERIAL_PORT, 
        baudrate=SERIAL_BAUD
    )

    turret = TurretController(
        arduino=arduino,
        pan_limits=(PAN_MIN, PAN_MAX),
        tilt_limits=(TILT_MIN, TILT_MAX),
        send_period=SERVO_SEND_PERIOD,
    )

    
    
    cams = StereoSystem(left=CAM_R_ID, right=CAM_L_ID)

    calib = load_final_calibration_data(CALIBRATION_OUTPUT_PATH)

    f = float(calib.Q[2, 3])
    B = float(-1.0 / calib.Q[3, 2])  # в мм
    last_send_t = 0.0
    last_sent_pan = None

    pan_deg = 90
    tilt_deg = 90

    print(f"Трекинг запущен. Нажмите '{EXIT_BUTTON}' для выхода.")

    frame_id = 0
    last_xR: Optional[int] = None
    z_prev: Optional[float] = None

    lost_count = 0
    last_show_l = None
    last_show_r = None
    begin: Optional[float] = None

    while True:
        frames = cams.capture_frame()
        if frames is None:
            continue

        frame_l = frames.camera_frame_l.frame
        frame_r = frames.camera_frame_r.frame

        rect_l = cv2.remap(frame_l, calib.map_l_x, calib.map_l_y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, calib.map_r_x, calib.map_r_y, cv2.INTER_LINEAR)

        rect_l_gray = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        rect_r_gray = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        pt, bbox, _ = det.get_target(frame_bgr=rect_l, return_debug=False)

        if pt is None:
            lost_count += 1
            if lost_count >= LOST_RESET_AFTER:
                z_prev = None
                last_xR = None

            # cv2.imshow("left", rect_l)
            # cv2.imshow("right", rect_r)
            cv2.imshow("left+right", cv2.hconcat([rect_l, rect_r]))

            k = cv2.waitKey(1) & 0xFF
            if k == ord(EXIT_BUTTON):
                break
            continue

        lost_count = 0
        xL, y = pt
        frame_id += 1

        xR = last_xR if last_xR is not None else xL

        # d = None
        # if frame_id % DEPTH_EVERY_N == 0 and 0 <= y < rect_l_gray.shape[0]:
        #     if DEPTH_METHOD == "sad":
        #         row_l = rect_l_gray[y]
        #         row_r = rect_r_gray[y]
        #         d = match_disparity_sad_1d(row_l, row_r, xL, max_disp=64, win=7)
        #     else:
        #         d = match_disparity_sgbm_strip(sgbm, rect_l_gray, rect_r_gray, xL, y)

        # if d is not None and d > 0:
        #     xR_new = int(round(xL - d))
        #     xR = xR_new
        #     last_xR = xR_new

        dz = det.deadzone_rect(frame_bgr=rect_l, deadzone_frac=(0.15, 0.15))
        inside = det.point_in_dz(pt_xy=pt, dz_xyxy=dz)
        vis = det.draw(frame_bgr=rect_l, det=bbox, pt=pt, dz=dz)

        # fx = P1[0, 0]
        # fy = P1[1, 1]
        # aim_err = compute_aim_error(pt=pt, dz_xyxy=dz, fx=fx, fy=fy)
        # # and aim_err.ey_ang is not None
        # if aim_err.ex_ang is not None:
        #     err_pan_deg = math.degrees(aim_err.ex_ang)

        #     if not inside:
        #         pan_deg += K_PAN * err_pan_deg
        #         pan_deg = max(PAN_MIN, min(PAN_MAX, pan_deg))

        #         now = time.time()
        #         new_pan = int(pan_deg)

        #         if now - last_send_t > SERVO_SEND_PERIOD and new_pan != last_sent_pan:
        #             ser.write(f"PAN={new_pan}\n".encode())
        #             last_send_t = now
        #             last_sent_pan = new_pan

        if inside:
            turret.set_light(enabled=True)
            now = time.time()
            if begin is not None:
                draw_timer(vis=vis, now=now, begin=begin)
            if begin is None:
                begin = time.time()
            elif abs(now - begin) > 1.5:
                turret.shoot()
                begin = None
        else:
            turret.set_light(enabled=False)
            begin = None

        

        color = (0, 255, 0) if inside else (0, 0, 255)
        if 0 <= y < rect_r.shape[0] and 0 <= xR < rect_r.shape[1]:
            cv2.circle(rect_r, (xR, y), 5, color, -1)

        z_raw = None
        # if d is not None and d > 0:
        #     z_candidate = (f * B) / float(d)
        #     if Z_MIN <= z_candidate <= Z_MAX:
        #         z_raw = z_candidate

        z_prev = update_depth_filtered(z_prev, z_raw, MAX_Z_STEP, EMA_ALPHA)

        

        # if z_prev is not None:
        #     cv2.putText(rect_r, f"Depth: {int(z_prev)} mm", (10, 30),
        #                 cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)

        draw_debug(
            vis=vis,
            inside=inside,
            pan=pan_deg,
            tilt=tilt_deg
        )
        
        cv2.imshow("left+right", cv2.hconcat([vis, rect_r]))
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord(EXIT_BUTTON):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking()
