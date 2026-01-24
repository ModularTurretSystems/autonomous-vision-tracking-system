from cv2.aruco import(
    CharucoBoard, 
    CharucoParameters, 
    DetectorParameters, 
    RefineParameters, 
    CharucoDetector,
    drawDetectedCornersCharuco,
    drawDetectedMarkers
)

from cv2.typing import MatLike, Scalar
from .types import CharucoDetectionResult, CharucoData, MarkerData
from typing import Sequence


class CharucoPattern:
    def __init__(
        self,
        charuco_board: CharucoBoard,
        *,
        charuco_params: CharucoParameters | None = None,
        detector_params: DetectorParameters | None = None,
        refine_params: RefineParameters | None = None,
    ) -> None:
        self.board = charuco_board
        self.detector = CharucoDetector(board=self.board, charucoParams=charuco_params, detectorParams=detector_params, refineParams=refine_params) #type: ignore


    #-------------------------------
    #========== DETECTION ==========
    #-------------------------------
    def detect_charuco_board(self, img: MatLike) -> CharucoDetectionResult:
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(img)

        found_charuco = charuco_corners is not None and charuco_ids is not None #type: ignore
        found_markers = marker_corners is not None and marker_ids is not None #type: ignore

        return CharucoDetectionResult(
            charuco=CharucoData(found=found_charuco, corners=charuco_corners, ids=charuco_ids),
            markers=MarkerData(found=found_markers, corners=marker_corners, ids=marker_ids)
        )
        

    def detect_charuco(self, img: MatLike) -> CharucoData:
        charuco_corners, charuco_ids, _, _ = self.detector.detectBoard(image=img)

        found = charuco_corners is not None and charuco_ids is not None #type: ignore
        return CharucoData(found=found, corners=charuco_corners, ids=charuco_ids)
    

    def detect_markers(self, img: MatLike) -> MarkerData:
        _, _, marker_corners, marker_ids = self.detector.detectBoard(image=img)

        found = marker_corners is not None and marker_ids is not None #type: ignore
        return MarkerData(found=found, corners=marker_corners, ids=marker_ids)


    #-------------------------------
    #========== DRAWING ==========
    #-------------------------------
    def draw_charuco_board(self, img: MatLike, res: CharucoDetectionResult, *, corner_color: Scalar | None = (255, 0, 0), border_color: Scalar | None = (255, 0, 0)) -> MatLike:
        self.draw_charuco(img=img, data=res.charuco, corner_color=corner_color)
        self.draw_markers(img=img, data=res.markers, border_color=border_color)

        return img


    def draw_charuco(self, img: MatLike, data: CharucoData, *, corner_color: Scalar | None = (255, 0, 0)) -> MatLike:
        if data.corners is not None: drawDetectedCornersCharuco(image=img, charucoCorners=data.corners, charucoIds=data.ids, cornerColor=corner_color) #type: ignore

        return img
    

    def draw_markers(self, img: MatLike, data: MarkerData, *, border_color: Scalar | None = (255, 0, 0)) -> MatLike:
        if data.corners is not None: drawDetectedMarkers(image=img, corners=data.corners, ids=data.ids, borderColor=border_color) #type: ignore

        return img
    

    def draw_corners(self, img: MatLike, res: CharucoDetectionResult, corner_color: Scalar | None = (255, 0, 0), border_color: Scalar | None = (255, 0, 0)) -> MatLike:
        self.draw_charuco(img=img, data=CharucoData(found=res.charuco.found, corners=res.charuco.corners, ids=None), corner_color=corner_color) #type: ignore
        self.draw_markers(img=img, data=MarkerData(found=res.markers.found, corners=res.markers.corners, ids=None), border_color=border_color) #type: ignore

        return img
    

    def draw_charuco_board_manual(
        self,
        img: MatLike,
        charuco_corners: MatLike | None = None,
        charuco_ids: MatLike | None = None,
        marker_corners: Sequence[MatLike] | None = None,
        marker_ids: MatLike | None = None,
        corner_color: Scalar | None = (255, 0, 0),
        border_color: Scalar | None = (255, 0, 0)
    ) -> MatLike:
        self.draw_charuco_board(
            img=img,
            res=CharucoDetectionResult(
                charuco=CharucoData(found=True, corners=charuco_corners, ids=charuco_ids), #type: ignore
                markers=MarkerData(found=True, corners=marker_corners, ids=marker_ids) #type: ignore
            ),
            corner_color=corner_color,
            border_color=border_color
        )

        return img
    