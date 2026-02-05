from typing import Optional, Tuple
from src.actuation import arduino
from src.actuation.arduino import ArduinoSerial

import time

class TurretController:
    def __init__(
        self,
        arduino: ArduinoSerial,
        pan_limits: Tuple[int, int] = (20, 160),
        tilt_limits: Tuple[int, int] = (30, 150),
        send_period: float = 0.05
        ) -> None:
        self.arduino = arduino
        self.pan_min, self.pan_max = pan_limits
        self.tilt_min, self.tilt_max = tilt_limits
        self.send_period = send_period

        self.last_send_t: float = 0.0
        self.last_pan:Optional[int] = None
        self.last_tilt:Optional[int] = None

        self.light_ON: bool = False

    # Servo
    def set_pan_tilt(
        self,
        pan: float,
        tilt: float
    ) -> None:
        pan_i = int(max(self.pan_min, min(self.pan_max, pan)))
        tilt_i = int(max(self.tilt_min, min(self.tilt_max, tilt)))

        now = time.time()
        if now - self.last_send_t < self.send_period:
            return

        if pan_i != self.last_pan:
            self.arduino.send(f"PAN={pan_i}")
            self.last_pan = pan_i

        if tilt_i != self.last_tilt:
            self.arduino.send(f"TILT={tilt_i}")
            self.last_tilt = tilt_i

        self.last_send_t = now

    # Light
    def set_light(
        self,
        enabled: bool
    ) -> None:
        if enabled and not self.light_ON:
            self.arduino.send("Light ON")
            self.light_ON = True
        elif not enabled and self.light_ON:
            self.arduino.send("Light OFF")
            self.light_ON = False
    
    # Shoot
    def shoot(self) -> None:
        self.arduino.send("Shoot")
        