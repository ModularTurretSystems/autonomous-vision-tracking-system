from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QMouseEvent, QPaintEvent, QPainter, QColor, QPixmap
from PySide6.QtCore import Signal, Qt


class TargetOverlay(QWidget):
    target_clicked = Signal(float, float)
    dragged = Signal(float, float)

    def __init__(self):
        super().__init__()

        self.setMinimumSize(200, 200)

        self.target_image = QPixmap("data/target.png")

        self.current_pos = (0.5, 0.5)
        self._dragging = False

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)

        w = self.width()
        h = self.height()

        painter.fillRect(self.rect(), QColor(0, 255, 0))


        # if self.target_image.isNull():
            # painter.drawPixmap(self.rect(), self.target_image)

        painter.drawPixmap(self.rect(), self.target_image)
        

        cx = int(self.current_pos[0] * w)
        cy = int(self.current_pos[1] * h)

        painter.setBrush(QColor(255, 0, 0))
        painter.drawEllipse(cx - 5, cy - 5, 10, 10)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._emit_click(event)

        if event.button() == Qt.MouseButton.RightButton:
            self._dragging = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            x, y = self._normalize(event)
            self.dragged.emit(x, y)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._dragging = False

    def _emit_click(self, ev: QMouseEvent) -> None:
        x, y = self._normalize(ev)
        self.target_clicked.emit(x, y)

    def _normalize(self, ev: QMouseEvent):
        x = ev.position().x() / self.width()
        y = ev.position().y() / self.height()

        x = max(0, min(1, x))
        y = max(0, min(1, y))

        return x, y
    
    def set_current(self, x: float, y: float):
        self.current_pos = (x, y)
        self.update()
