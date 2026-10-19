from PySide6.QtCore import QObject, Signal


################################################################################

import serial
import serial.tools.list_ports
from serial import Serial
import time

from src.targeting.types import Angles
from src.targeting.aiming import AimingCalculator



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


def send_angle(arduino: Serial, angle: int):
    print(angle)
    if arduino and arduino.is_open:
        try:
            arduino.write(f"{angle}\n".encode())

            # time.sleep(0.05)
            while arduino.in_waiting:
                response = arduino.readline().decode().strip()
                if response.startswith("ANGLE_SET:"):
                    current = response.split(":")[1]
                    print(f"Arduino approved angle {current}")
        
        except Exception as e:
            print(f"Error {e}")


IMAGE_SIZE = (1280, 720)
FOV = 65
AOV = 37
center_img = (IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2)
RANGE = (0, 180)

calclulator = AimingCalculator(image_size=IMAGE_SIZE, fov=FOV, aov=AOV, org=center_img, pan_range=RANGE, tilt_range=RANGE)

inital_angles = Angles(pan=90, tilt=90)

arduino = connect_arduino()
if arduino is None: exit()

################################################################################


def norm_to_pixels(x: float, y: float, size: tuple[int, int]):
    w, h = size
    return (x * w, y * h)


class TurretController(QObject):

    position_changed = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.current = (0.5, 0.5)
        send_angle(arduino=arduino, angle=90) # type: ignore


    def aim_to(self, x: float, y: float):
        # print(f"[CLICK AIM] -> {x:.2f}, {y:.2f}")
        self.current = (x, y)

        self.position_changed.emit(x, y)
        self.apply()

    def drag_control(self, x: float, y: float):
        # print(f"[DRAG] -> {x:.2f}, {y:.2f}")

        self.current = (x, y)

        self.position_changed.emit(x, y)
        self.apply()

    def apply(self):
        target_px = norm_to_pixels(self.current[0], self.current[1], IMAGE_SIZE)
        # print(target_px)
        angles = calclulator.compute_pan_tilt_angles(current=inital_angles, target=target_px)

        send_angle(arduino=arduino,angle=int(angles.pan)) # type: ignore
