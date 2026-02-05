from typing import Optional

import numpy as np
import cv2

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
