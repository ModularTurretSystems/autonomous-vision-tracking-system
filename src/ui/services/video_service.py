from PySide6.QtCore import QThread, Signal
import numpy as np

from src.camera import Camera


class VideoService(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera
        self._running = True

    def run(self):
        while(self._running):
            frame = self.camera.capture_frame()

            if not frame.success:
                continue

            self.frame_ready.emit(frame.frame)

    def stop(self):
        self._running = False
        self.wait()
