import sys
from PyQt5 import QtWidgets

from source.main.select_window import MainWindow
from source.main.video_loader import VideoLoader


class Controller:
    def __init__(self):
        pass

    def show_video_loader(self):
        self.video_loader = VideoLoader(self.window.video_path)
        # self.video_loader.switch_window.connect(self.show_main_window)
        self.video_loader.show()

    def show_main_window(self):
        self.window = MainWindow()
        self.window.switch_window.connect(self.show_video_loader)
        self.window.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    controller = Controller()
    controller.show_main_window()

    sys.exit(app.exec_())
