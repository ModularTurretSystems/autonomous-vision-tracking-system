from src.camera.stereo_camera import StereoCapture
import cv2

def collect_images() -> None:
        

    with StereoCapture(left_id=0, right_id=1, width=640, height=480) as stereo:

        img_count = 0
        
        while True:
            frames = stereo.get_frames()

            frame_l = frames.left
            frame_r = frames.right
            

            cv2.putText(
                img=frame_l,
                text=f"Photos: {img_count}",
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
                img_count += 1
                stereo.save_pairs(frame_left=frame_l, frame_right=frame_r, count=img_count)

    cv2.destroyAllWindows()

   