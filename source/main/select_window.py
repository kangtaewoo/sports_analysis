from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from source.ui.select_window_ui import Ui_SelectWindow


class MainWindow(QMainWindow, Ui_SelectWindow):

    switch_window = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(lambda :self.btn_clicked(self.pushButton))
        self.pushButton_2.clicked.connect(lambda :self.btn_clicked(self.pushButton_2))
        self.pushButton_3.clicked.connect(lambda :self.btn_clicked(self.pushButton_3))

    def btn_clicked(self, button):
        if button.text() == "축구":
            self.video_path = "../../../videos/test1.mp4"
        elif button.text() == "농구":
            pass
        elif button.text() == "핸드볼":
            pass

        self.switch_window.emit()
