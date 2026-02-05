from typing import List, Literal, Optional, Tuple, Any, Dict, Union
from cv2.typing import MatLike
from .types import to_numpy
from ultralytics.models.yolo import YOLO

import numpy as np
import cv2

from src.tracking.types import Aim, BBoxDet

# New types for changing mods of targeting
PickMode = Literal["closest_center", "highest_conf", "largest"]
AimPoint = Literal["torso", "bbox_center", "head"]


class YoloTargetDetector:

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        aim_point: AimPoint = "torso",
        pick_mode: PickMode = "closest_center",
        conf_th: float = 0.45,
        iou_th: float = 0.5,  # remove duplicates
        classes: Optional[
            List[float]
        ] = None,  # None = all classes, but [0] is only person
        device: Optional[str] = None,  # cpu, cuda:0, etc.
        ema_alpha: float = 0.35,  # 0...1 higher = stronger smoothing
    ) -> None:
        self.model = YOLO(model_path)

        self.conf_th = conf_th
        self.iou_th = iou_th
        self.classes = classes if classes is not None else [0]
        self.device = device

        self.aim_point = aim_point
        self.pick_mode = pick_mode
        self.ema_alpha = ema_alpha

        self._ema_xy: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._ema_xy = None

    def detect(self, frame_bgr: np.ndarray) -> List[BBoxDet]:
        res = self.model.predict(
            source=frame_bgr,
            conf=self.conf_th,
            iou=self.iou_th,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = to_numpy(boxes.xyxy).astype(np.float32)
        clss = to_numpy(boxes.cls).astype(int)
        conf = to_numpy(boxes.conf).astype(float)

        dets: List[BBoxDet] = []

        for box in range(len(xyxy)):
            dets.append(
                BBoxDet(xyxy=xyxy[box], conf=float(conf[box]), cls=int(clss[box]))
            )

        return dets

    def select_primary(
        self, dets: List[BBoxDet], frame_shape: Tuple[int, int, int]
    ) -> Optional[BBoxDet]:
        if not dets:
            return None

        h, w = frame_shape[:2]

        fx, fy = w * 0.5, h * 0.5

        if self.pick_mode == "highest_conf":
            return max(dets, key=lambda d: d.conf)

        if self.pick_mode == "largest":
            return max(dets, key=lambda d: d.area)

        # closest_center (The Euclidean distance, but without sqrt, cause code will take bbox)
        return min(dets, key=lambda d: (d.center_x - fx) ** 2 + (d.center_y - fy) ** 2)

    def aim_from_bbox(self, det: BBoxDet) -> Aim:
        x1, y1, x2, y2 = det.xyxy

        if self.aim_point == "bbox_center":
            return Aim(x=float((x2 + x1) * 0.5), y=float((y2 + y1) * 0.5))

        if self.aim_point == "head":
            return Aim(x=float((x2 + x1) * 0.5), y=float((y2 - y1) * 0.18 + y1))
        # torso
        return Aim(x=float((x1 + x2) * 0.5), y=float(y1 + 0.45 * (y2 - y1)))

    def get_target(
        self, frame_bgr: np.ndarray, return_debug: bool = False
    ) -> Tuple[Optional[Tuple[int, int]], Optional[BBoxDet], Optional[Dict[str, Any]]]:
        dets = self.detect(frame_bgr=frame_bgr)
        det = self.select_primary(dets=dets, frame_shape=frame_bgr.shape)

        if det is None:
            return None, None, ({"dets": 0} if return_debug else None)

        raw_xy = np.array(self.aim_from_bbox(det=det).get_xy, dtype=np.float32)

        if self._ema_xy is None:
            self._ema_xy = raw_xy
        else:
            a = self.ema_alpha
            self._ema_xy = a * self._ema_xy + (1.0 - a) * raw_xy

        pt = int(round(float(self._ema_xy[0]))), int(round(float(self._ema_xy[1])))

        dbg = None
        if return_debug:
            dbg = {
                "dets": len(dets),
                "picked_conf": det.conf,
                "pick_mode": self.pick_mode,
                "aim_point": self.aim_point,
                "raw_xy": (float(raw_xy[0]), float(raw_xy[1])),
                "ema_xy": (float(self._ema_xy[0]), float(self._ema_xy[1])),
            }

        return pt, det, dbg

    @staticmethod
    def deadzone_rect(
        frame_bgr: MatLike, deadzone_frac: Tuple[float, float] = (0.12, 0.12)
    ) -> Tuple[int, int, int, int]:  # x1, y1, x2, y2
        h, w = frame_bgr.shape[:2]

        # fr = np.asarray(deadzone_frac, dtype=np.float32).reshape(-1)
        # dx_frac = float(fr[0])
        # dy_frac = float(fr[1])

        cx, cy = w * 0.5, h * 0.5
        hx = deadzone_frac[0] * w * 0.5
        hy = deadzone_frac[1] * h * 0.5

        x1 = int(round(cx - hx))
        y1 = int(round(cy - hy))
        x2 = int(round(cx + hx))
        y2 = int(round(cy + hy))
        return x1, y1, x2, y2

    @staticmethod
    def point_in_dz(pt_xy: Tuple[int, int], dz_xyxy: Tuple[int, int, int, int]) -> bool:
        x, y = pt_xy
        x1, y1, x2, y2 = dz_xyxy
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    @staticmethod
    def draw(
        frame_bgr: np.ndarray,
        det: Optional[BBoxDet],
        pt: Optional[Tuple[int, int]],
        dz: Tuple[int, int, int, int],
    ) -> np.ndarray:
        out = frame_bgr.copy()

        # deadzone
        x1, y1, x2, y2 = dz
        cv2.rectangle(
            img=out, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=2
        )

        if det is not None:
            bx1, by1, bx2, by2 = det.xyxy.astype(int)
            cv2.rectangle(
                img=out, pt1=(bx1, by1), pt2=(bx2, by2), color=(0, 255, 0), thickness=2
            )
            cv2.putText(
                img=out,
                text=f"person: {det.conf:.2f}",
                org=(bx1, max(0, by1 - 8)),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=2,
            )

            # aim point
            if pt is not None:
                cv2.circle(
                    img=out, center=pt, radius=6, color=(0, 255, 255), thickness=-1
                )

        return out
