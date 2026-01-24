import cv2
from cv2 import aruco
from cv2.aruco import CharucoBoard, getPredefinedDictionary
from src.calibration.patterns.charuco import CharucoPattern

from src.camera.camera import Camera

SIZE = (5, 7)
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
DICTIONARY = aruco.DICT_5X5_100

PARAMS = [
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720
]

RESOLUTION = (1280, 720)

dictionary = getPredefinedDictionary(dict=DICTIONARY)

board = CharucoBoard(size=SIZE, squareLength=SQUARE_LENGTH, markerLength=MARKER_LENGTH, dictionary=dictionary)

refine_params = aruco.RefineParameters(checkAllOrders=True)
detector_params = aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE


pattern = CharucoPattern(charuco_board=board, detector_params=detector_params, refine_params=refine_params)

cam = Camera(camera_id=0, apiPreference=cv2.CAP_MSMF, params=PARAMS)


while(1):
    ret, frame = cam.capture_frame()

    res = pattern.detect_charuco_board(img=frame)
    
    pattern.draw_charuco_board(img=frame, res=res)
    # pattern.draw_corners(img=frame, res=res)
    # pattern.draw_charuco_board_manual(img=frame, marker_corners=res.markers.corners)



        # n, charuco_corners_interpolated, charuco_ids_interpolated = aruco.interpolateCornersCharuco( # Доп интерполяция
        #     markerCorners=marker_corners, 
        #     markerIds=marker_ids, 
        #     image=frame_interpolated, 
        #     board=board
        # )
        # aruco.drawDetectedCornersCharuco(image=frame_interpolated, charucoCorners=charuco_corners_interpolated, cornerColor=(0, 0, 255))


    cv2.imshow(winname="img", mat=frame)

    if cv2.waitKey(delay=1) == ord('q'):
        break
