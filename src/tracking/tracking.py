from pathlib import Path
import re
from typing import Any, Dict, Optional, Literal

import cv2
import numpy as np
import serial
import time
import math

from src.tracking.errors import compute_aim_error
from src.tracking.yolo_tracker import YoloTargetDetector
from src.utils.types import CFG
from src.vision.stereo.system import StereoSystem


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


def load_final_calibration_data(npz_path: str | Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "map_l": (
            data["map_l_x"].astype(np.float32),
            data["map_l_y"].astype(np.float32),
        ),
        "map_r": (
            data["map_r_x"].astype(np.float32),
            data["map_r_y"].astype(np.float32),
        ),
        "Q": data["Q"].astype(np.float32),
        "P1": data["P1"].astype(np.float32),
    }


def match_disparity_sad_1d(
    row_l: np.ndarray,
    row_r: np.ndarray,
    xL: int,
    max_disp: int = 64,
    win: int = 7,
) -> Optional[float]:
    """
    Возвращает disparity d (float) или None.
    disparity = xL - xR, d > 0
    """
    half = win // 2
    w = row_r.shape[0]

    if xL - half < 0 or xL + half >= w:
        return None

    wl = row_l[xL - half : xL + half + 1].astype(np.int16)

    best_d = None
    best_cost = 1e18

    for d in range(1, max_disp + 1):
        xR = xL - d
        if xR - half < 0 or xR + half >= w:
            continue

        wr = row_r[xR - half : xR + half + 1].astype(np.int16)
        cost = int(np.sum(np.abs(wl - wr)))

        if cost < best_cost:
            best_cost = cost
            best_d = d

    if best_d is None:
        return None

    return float(best_d)


def match_disparity_sgbm_strip(
    stereo: cv2.StereoSGBM,
    img_l_gray: np.ndarray,
    img_r_gray: np.ndarray,
    xL: int,
    y: int,
    strip_half_h: int = 6,
    roi_margin_x: int = 120,
) -> Optional[float]:
    """
    Считает disparity на узкой горизонтальной полосе вокруг y.
    Возвращает disparity d (float) или None.
    """
    h, w = img_l_gray.shape[:2]
    y0 = max(0, y - strip_half_h)
    y1 = min(h, y + strip_half_h + 1)

    x0 = max(0, xL - roi_margin_x)
    x1 = min(w, xL + roi_margin_x)

    roi_w = x1 - x0
    num_disp = stereo.getNumDisparities()
    block = stereo.getBlockSize()
    if roi_w - num_disp <= (block // 2) + 1:
        x0, x1 = 0, w
        roi_w = w
        if roi_w - num_disp <= (block // 2) + 1:
            return None

    roi_l = img_l_gray[y0:y1, x0:x1]
    roi_r = img_r_gray[y0:y1, x0:x1]

    disp = stereo.compute(roi_l, roi_r).astype(np.float32) / 16.0  # pixels

    x_roi = xL - x0
    y_roi = y - y0

    dh, dw = disp.shape
    if not (0 <= y_roi < dh and 0 <= x_roi < dw):
        return None

    win = 5
    half = win // 2
    if x_roi - half < 0 or x_roi + half >= dw or y_roi - half < 0 or y_roi + half >= dh:
        return None

    patch = disp[y_roi - half : y_roi + half + 1, x_roi - half : x_roi + half + 1]
    valid = patch[patch > 0.5]  # 0/отрицательное у SGBM — невалид
    if valid.size == 0:
        return None

    d = float(np.median(valid))
    return d


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

    cams = StereoSystem(left=CAM_R_ID, right=CAM_L_ID)

    calib = load_final_calibration_data(CALIBRATION_OUTPUT_PATH)
    map_lx, map_ly = calib["map_l"]
    map_rx, map_ry = calib["map_r"]
    Q = calib["Q"]
    P1 = calib["P1"]

    f = float(Q[2, 3])
    B = float(-1.0 / Q[3, 2])  # в мм
    last_send_t = 0.0
    last_sent_pan = None
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=SERIAL_BAUD,
        timeout=0.01,
    )

    time.sleep(2)

    pan_deg = 90
    tilt_deg = 90

    sgbm = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=64,  # кратно 16
        blockSize=5,
        P1=8 * 1 * 5 * 5,
        P2=32 * 1 * 5 * 5,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=16,
        disp12MaxDiff=1,
    )

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

        rect_l = cv2.remap(frame_l, map_lx, map_ly, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map_rx, map_ry, cv2.INTER_LINEAR)

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
            cv2.imshow("хуйня", cv2.hconcat([rect_l, rect_r]))

            k = cv2.waitKey(1) & 0xFF
            if k == ord(EXIT_BUTTON):
                break
            continue

        lost_count = 0
        xL, y = pt
        frame_id += 1

        xR = last_xR if last_xR is not None else xL

        d = None
        if frame_id % DEPTH_EVERY_N == 0 and 0 <= y < rect_l_gray.shape[0]:
            if DEPTH_METHOD == "sad":
                row_l = rect_l_gray[y]
                row_r = rect_r_gray[y]
                d = match_disparity_sad_1d(row_l, row_r, xL, max_disp=64, win=7)
            else:
                d = match_disparity_sgbm_strip(sgbm, rect_l_gray, rect_r_gray, xL, y)

        if d is not None and d > 0:
            xR_new = int(round(xL - d))
            xR = xR_new
            last_xR = xR_new

        dz = det.deadzone_rect(frame_bgr=rect_l, deadzone_frac=(0.15, 0.15))
        inside = det.point_in_dz(pt_xy=pt, dz_xyxy=dz)
        vis = det.draw(frame_bgr=rect_l, det=bbox, pt=pt, dz=dz)

        fx = P1[0, 0]
        fy = P1[1, 1]
        aim_err = compute_aim_error(pt=pt, dz_xyxy=dz, fx=fx, fy=fy)
        # and aim_err.ey_ang is not None
        if aim_err.ex_ang is not None:
            err_pan_deg = math.degrees(aim_err.ex_ang)

            if not inside:
                pan_deg += K_PAN * err_pan_deg
                pan_deg = max(PAN_MIN, min(PAN_MAX, pan_deg))

                now = time.time()
                new_pan = int(pan_deg)

                if now - last_send_t > SERVO_SEND_PERIOD and new_pan != last_sent_pan:
                    ser.write(f"PAN={new_pan}\n".encode())
                    last_send_t = now
                    last_sent_pan = new_pan
                
                

        if inside:
            ser.write(b"Light ON\n")
            n = time.time()
            if begin is not None:
                cv2.putText(
                    vis,
                    f"Sec={abs(n-begin):.2f}",
                    (10, 120),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            if begin is None:
                begin = time.time()
            elif abs(n - begin) > 1.5:
                ser.write(b"Shoot\n")
                begin = None
        else:
            ser.write(b"Light OFF\n")
            begin = None

        cv2.putText(
            vis,
            f"PAN={int(pan_deg)} TILT={int(tilt_deg)}",
            (10, 90),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        color = (0, 255, 0) if inside else (0, 0, 255)
        if 0 <= y < rect_r.shape[0] and 0 <= xR < rect_r.shape[1]:
            cv2.circle(rect_r, (xR, y), 5, color, -1)

        z_raw = None
        if d is not None and d > 0:
            z_candidate = (f * B) / float(d)
            if Z_MIN <= z_candidate <= Z_MAX:
                z_raw = z_candidate

        z_prev = update_depth_filtered(z_prev, z_raw, MAX_Z_STEP, EMA_ALPHA)

        cv2.putText(
            vis,
            f"inside_deadzone: {inside}",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        # if z_prev is not None:
        #     cv2.putText(rect_r, f"Depth: {int(z_prev)} mm", (10, 30),
        #                 cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)

        # cv2.imshow("left", vis)
        # cv2.imshow("right", rect_r)
        cv2.imshow("хуйня", cv2.hconcat([vis, rect_r]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord(EXIT_BUTTON):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking()
