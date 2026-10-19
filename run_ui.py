import cv2

from src.ui import run

from src.camera import Camera


CAM_ID = 0
PARAMS = (
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720,
)


def main():
    camera = Camera(camera_id=CAM_ID, apiPreference=cv2.CAP_V4L2, params=PARAMS)
    run(camera)

if __name__ == "__main__":
    main()
