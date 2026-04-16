from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import(
    QMainWindow, 
    QWidget, 
    QGridLayout
)

from src.ui.widgets import VideoWidget, TargetOverlay
from src.ui.services import VideoService
from src.ui.controllers import TurretController

from src.camera import Camera


class MainWindow(QMainWindow):
    def __init__(self, camera: Camera):
        super().__init__()

        self.setWindowTitle("Turret Control UI")

        # Widgets
        self.video_widget = VideoWidget()
        self.target_widget = TargetOverlay()

        # Main layout
        container = QWidget()
        layout = QGridLayout()

        layout.addWidget(self.video_widget, 1, 0, 2, 2)
        layout.addWidget(self.target_widget, 0, 2)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 2)

        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 1)

        container.setLayout(layout)
        self.setCentralWidget(container)

        # Controller
        self.controller = TurretController()

        # Video
        self.video_service = VideoService(camera)
        self.video_service.frame_ready.connect(self.video_widget.update_frame)

        # Mouse -> controller
        self.video_widget.click_aim.connect(self.controller.aim_to)

        self.target_widget.target_clicked.connect(self.controller.aim_to)
        self.target_widget.dragged.connect(self.controller.drag_control)

        self.controller.position_changed.connect(self.target_widget.set_current)

        self.video_service.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.video_service.stop()
        event.accept()
