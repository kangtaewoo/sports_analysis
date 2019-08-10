from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class VideoLabel(QLabel):
    # release = pyqtSignal(tuple)

    def __init__(self, view):
        super().__init__(view)
        #선수 선택
        self.flag = None
        self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = None
        # self.rect = None

    def mousePressEvent(self, event):
        if self.flag is True:
            self.origin = self.mapFromParent(event.pos())
            self.rubberband.setGeometry(QRect(self.origin, QSize()))
            self.rubberband.show()
            QWidget.mousePressEvent(self, event)
        # else:
        #     event.ignore()


    def mouseMoveEvent(self, event):
        if self.rubberband.isVisible():
            self.rubberband.setGeometry(QRect(self.origin, event.pos()).normalized())
            QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        # if self.rubberband.isVisible():
        size = self.rubberband.size()
        self.rect = (float(self.origin.x()), float(self.origin.y()), float(size.width()), float(size.height()))