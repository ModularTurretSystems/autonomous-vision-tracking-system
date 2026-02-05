import time

from typing import Optional

from serial import Serial


class ArduinoSerial:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.01,
        init_delay: float = 2,
    ) -> None:

        self.ser = Serial(port=port, baudrate=baudrate, timeout=timeout)

        time.sleep(init_delay)

    def send(self, cmd: str) -> None:
        self.ser.write((cmd + "\n").encode())

    def close(self) -> None:
        self.ser.close()
