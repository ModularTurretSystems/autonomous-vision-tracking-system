from cv2.typing import MatLike

import cv2

def draw_debug(
    vis: MatLike,
    inside: bool,
    pan: int,
    tilt: int
) -> None:

    cv2.putText(
        vis,
        f"PAN={pan} TILT={tilt}",
        (10, 90),
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        vis,
        f"inside_deadzone: {inside}",
        (10, 30),
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

def draw_timer(
    vis: MatLike,
    now: float,
    begin: float
) -> None:
    cv2.putText(
        vis,
        f"Sec={abs(now-begin):.2f}",
        (10, 120),
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
