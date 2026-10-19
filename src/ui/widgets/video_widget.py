from PySide6.QtCore import QSize, Signal, Qt
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
import cv2
from cv2.typing import MatLike


class VideoWidget(QLabel):
    click_aim = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setText("No video")
        self.setScaledContents(True)

        self.setMouseTracking(True)

    def update_frame(self, frame: MatLike):
        # BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        image = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        self.setPixmap(QPixmap.fromImage(image))

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            x = ev.position().x() / self.width()
            y = ev.position().y() / self.height()

            x = max(0, min(1, x))
            y = max(0, min(1, y))

            self.click_aim.emit(x, y)


    def minimumSizeHint(self) -> QSize:
        return QSize(160, 90)
