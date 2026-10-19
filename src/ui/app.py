import sys
from PySide6.QtWidgets import QApplication

from src.ui import MainWindow
from src.camera import Camera


def run(camera: Camera):
    app = QApplication(sys.argv)

    window = MainWindow(camera)
    window.resize(1280, 720)
    window.show()

    sys.exit(app.exec())
