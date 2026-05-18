from src.calibration.mono.types import MonoCalibrationResult
from src.targeting.aiming import AimingCalculator
import cv2
from src.tracking.tracker import Tracker
from src.tracking.config import TrackerConfig
from src.tracking.constants import TrackerModel
from src.targeting.selector import TargetSelector
from src.targeting.constants import SelectorStrategy
from src.targeting.types import Angles
from src.camera.camera import Camera

from src.calibration.persistence.npz import NpzCalibrationStorage
from src.calibration.stereo.types import StereoCalibrationResult

import serial
import serial.tools.list_ports
import time
import threading
from serial import Serial

ARDUINO_PORT = None
ARDUINO_BAUD = 9600


def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description or "CH340" in port.description:
            return port.device
    return None


def connect_arduino():
    port = ARDUINO_PORT
    if port is None:
        port = find_arduino_port()

    if port:
        try:
            arduino = serial.Serial(port=port, baudrate=ARDUINO_BAUD, timeout=1)
            time.sleep(2)
            print("Arduino connected")

            # while(True):
            #     line = arduino.readline().decode().strip()
            #     if line == "ARDUINO_READY":
            #         print("Arduino ready")
            #         break
            
            return arduino
        except Exception as e:
            print(f"Error {e}")
            return None
    else:
        print("Arduino doesn't found")
        return None

def send_angles_relative(
    arduino: Serial,
    pan: int,
    tilt: int,
):
    print(f"[REL] PAN={pan}, TILT={tilt}")

    if arduino and arduino.is_open:
        try:
            command = (
                f"PAN={pan}, "
                f"TILT={tilt}\n"
            )

            arduino.write(command.encode())

        except Exception as e:
            print(f"Error: {e}")

def send_angles_absolute(
    arduino: Serial,
    pan: int,
    tilt: int,
):
    print(f"[ABS] PAN={pan}, TILT={tilt}")

    if arduino and arduino.is_open:
        try:
            command = (
                f"SETPAN={pan}, "
                f"SETTILT={tilt}\n"
            )

            arduino.write(command.encode())

        except Exception as e:
            print(f"Error: {e}")


def shoot(arduino: Serial):
    print("Shoot")

    if arduino and arduino.is_open:
        try:
            pass
            # arduino.write(f"P{pan} T{tilt}\n".encode())
            # arduino.write(f"PAN={pan}, TILT={tilt}\n".encode())
            arduino.write("fire\n".encode())

            # while arduino.in_waiting:
            #     response = arduino.readline().decode().strip()
            #     print(f"[ARDUINO] {response}")

        except Exception as e:
            print(f"Error {e}")


def stop(arduino: Serial):
    print("Stop")

    if arduino and arduino.is_open:
        try:
            pass
            # arduino.write(f"P{pan} T{tilt}\n".encode())
            # arduino.write(f"PAN={pan}, TILT={tilt}\n".encode())
            arduino.write("stop\n".encode())

            # while arduino.in_waiting:
            #     response = arduino.readline().decode().strip()
            #     print(f"[ARDUINO] {response}")

        except Exception as e:
            print(f"Error {e}")

def start(arduino: Serial):
    print("start")

    if arduino and arduino.is_open:
        try:
            pass
            # arduino.write(f"P{pan} T{tilt}\n".encode())
            # arduino.write(f"PAN={pan}, TILT={tilt}\n".encode())
            arduino.write("start\n".encode())

            # while arduino.in_waiting:
            #     response = arduino.readline().decode().strip()
            #     print(f"[ARDUINO] {response}")

        except Exception as e:
            print(f"Error {e}")
                

MODEL = TrackerModel.YOLO26N
CLASSES = 0
RANGE = (0, 180)
WITH_REID = True

IMAGE_SIZE = (1280, 720)
CAM_ID = 0
PARAMS = (
    cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0],
    cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1]
)

FOV = 65
AOV = 37
# FOV = 45
# AOV = 37
config = TrackerConfig(model=MODEL, classes=CLASSES, device="cuda:0")
tracker = Tracker(config=config)

center_img = (IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2)
selector = TargetSelector(strategy=SelectorStrategy.CLOSEST_TO_CENTER, center=center_img)
center_img = int(center_img[0]), int(center_img[1])

# data = NpzCalibrationStorage(StereoCalibrationResult).load("data/npz/stereo.npz")
data = NpzCalibrationStorage(MonoCalibrationResult).load("data/RESULT.npz")
# data_tilt = NpzCalibrationStorage(MonoCalibrationResult).load("data/npz_stereo/left.npz")
# center_img = int(data.left_camera_matrix[0][2]), int(data.left_camera_matrix[1][2])
# print(center_img)

calclulator = AimingCalculator(image_size=IMAGE_SIZE, camera_matrix=data.camera_matrix, pan_range=RANGE, tilt_range=RANGE)
# calclulator_tilt = AimingCalculator(image_size=IMAGE_SIZE, camera_matrix=data_tilt.camera_matrix, pan_range=RANGE, tilt_range=RANGE)
# calclulator = AimingCalculator(image_size=IMAGE_SIZE, fov=FOV, aov=AOV, org=center_img, pan_range=RANGE, tilt_range=RANGE)
# calclulator = AimingCalculator(image_size=IMAGE_SIZE, fov=55, aov=37, org=center_img)

cam = Camera(camera_id=CAM_ID, apiPreference=cv2.CAP_MSMF, params=PARAMS)
# cam_tilt = Camera(camera_id=CAM_ID_TILT, apiPreference=cv2.CAP_MSMF, params=PARAMS)

# inital_angles = Angles(pan=90, tilt=90)

arduino = connect_arduino()
if arduino is None:
    exit()

PAN_DEFAULT = 0
TILT_DEFAULT = 0

DEAD_ANGLE = 5
SEND_INTERVAL = 0.2

START_DELAY = 2
SHOT_INTERVAL = 1.5
STOP_DELAY = 1.0

last_send_time = 0
last_shot_time = 0
last_detection_time = 0

flywheels_start_time = 0
flywheels_active = False

while True:
    cam_frame = cam.capture_frame()

    if not cam_frame.success:
        break

    frame = cam_frame.frame

    tracked_objects = tracker.track(frame=frame)
    selected_objects = selector.select(objects=tracked_objects)

    now = time.time()

    if len(selected_objects) != 0:

        obj = selected_objects[0]

        # Одна камера теперь считает сразу и pan и tilt
        angles = calclulator.compute_pan_tilt_angles(
            current=Angles(pan=0, tilt=0),
            target=obj.center
        )

        last_detection_time = now

        # ---- Управление маховиками ----
        if not flywheels_active:
            start(arduino)

            flywheels_active = True
            flywheels_start_time = now

            # Чтобы после разгона сразу можно было стрелять
            last_shot_time = now - SHOT_INTERVAL + START_DELAY

            print("Flywheels START")

        # ---- Выстрел ----
        if now - last_shot_time >= SHOT_INTERVAL:

            if now - flywheels_start_time >= START_DELAY:
                shoot(arduino)

                last_shot_time = now

                print("🔫 SHOT")

        # ---- Отправка углов ----
        if now - last_send_time >= SEND_INTERVAL:

            pan_angle = int(angles.pan)
            tilt_angle = int(angles.tilt)

            # Фильтр дрожания
            if (
                abs(pan_angle) >= DEAD_ANGLE
                or abs(tilt_angle) >= DEAD_ANGLE
            ):
                send_angles_relative(
                    arduino=arduino,
                    pan=pan_angle,
                    tilt=tilt_angle
                )

                print(f"PAN={pan_angle}, TILT={tilt_angle}")

                last_send_time = now

        # ---- Визуализация ----
        pt1, pt2, center = obj.pts_to_int()

        cv2.rectangle(
            img=frame,
            pt1=pt1,
            pt2=pt2,
            color=(0, 255, 0),
            thickness=1
        )

        cv2.circle(
            img=frame,
            center=center,
            radius=5,
            color=(0, 0, 255),
            thickness=-1
        )

        cv2.circle(
            img=frame,
            center=center_img,
            radius=5,
            color=(0, 0, 255),
            thickness=-1
        )

        cv2.arrowedLine(
            img=frame,
            pt1=center_img,
            pt2=center,
            color=(0, 0, 255),
            thickness=2,
            tipLength=0.1
        )

        text = (
            f"pan={round(angles.pan, 2)}, "
            f"tilt={round(angles.tilt, 2)}"
        )

        cv2.putText(
            img=frame,
            text=text,
            org=(center[0] - 50, center[1] - 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=2
        )

        text = f"class={obj.cls_name}"

        cv2.putText(
            img=frame,
            text=text,
            org=(center[0] - 50, center[1] + 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 0, 0),
            thickness=2
        )

        text = f"id={obj.id}"

        cv2.putText(
            img=frame,
            text=text,
            org=(center[0] - 50, center[1] + 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 0, 0),
            thickness=2
        )

    # ---- Отключение маховиков ----
    if flywheels_active and (now - last_detection_time >= STOP_DELAY):

        stop(arduino)

        flywheels_active = False

        print("Flywheels STOP")

    cv2.imshow(winname="res", mat=frame)

    if cv2.waitKey(delay=1) == ord('q'):
        break

cv2.destroyAllWindows()

send_angles_absolute(arduino=arduino, pan=90, tilt=100)
stop(arduino)

if arduino:
    arduino.close()
