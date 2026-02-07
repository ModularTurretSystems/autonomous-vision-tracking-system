import numpy as np
import cv2

from numpy.lib.stride_tricks import sliding_window_view

from typing import Optional, Tuple
from numpy.typing import NDArray


def sad_match_1d(
    row_left: NDArray[np.floating],
    row_right: NDArray[np.floating],
    x_left: int,
    windows_size: int,    
    max_dispatity: int     
) -> Optional[Tuple[float, int]]:
    half = windows_size // 2
    windows_size = 2 * half + 1

    length = row_left.shape[0]
    if x_left + half >= length or x_left - half < 0:
        return None
    
    w_left = row_left[x_left - half : x_left + half + 1]

    r_start = max(0, x_left - max_dispatity - half)
    r_end = min(row_right.shape[0], x_left + half)

    region = row_right[r_start : r_end + 1]

    candidate_windows = sliding_window_view(x=region, window_shape=windows_size)
    costs = np.sum(np.abs(candidate_windows - w_left), axis=1)

    min_idx = int(np.argmin(costs))
    x_right = r_start + min_idx + half

    disparity = float(x_left - x_right)
    
    return disparity, x_right


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
