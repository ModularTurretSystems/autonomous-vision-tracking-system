from src.vision.stereo.calibration import StereoCapture
from src.vision.stereo.system import StereoSystem
import cv2

def collect_images() -> None:
    stereo_system = StereoSystem(left=0, right=1)
    save_dir = "data/calibration_images"

    with StereoCapture(stereo_system=stereo_system, save_dir=save_dir) as stereo:
        while True:
            stereo_frame = stereo.capture_frame()

            frame_l = stereo_frame.frame_l
            frame_r = stereo_frame.frame_r

            cv2.putText(
                img=frame_l,
                text=f"Photos: {stereo.get_number_of_frames()}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),      
                thickness=2
            )


            cv2.putText(
                img=frame_r,
                text="Press 's' to save",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 255, 255), 
                thickness=2
            )

            cv2.putText(
                img=frame_r,
                text="Press 'q' to exit",
                org=(10, 470),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 0, 255),    
                thickness=2
            )

            cv2.imshow(winname="Left", mat=frame_l)
            cv2.imshow(winname="Right", mat=frame_r)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                stereo.save_frame(frame=stereo_frame)

    cv2.destroyAllWindows()

   