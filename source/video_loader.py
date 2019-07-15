import cv2
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from source.ui.video_loader_ui import Ui_MainWindow
from source.video_preprocessor import Video_Preprocessor

class VideoThread(QThread):

    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.run_flag = False
        self.cap = cv2.VideoCapture("../../videos/test1.mp4")
        self.pre = Video_Preprocessor()
        self.count = 0

    def __del__(self):
        self.cap.release()

    def run(self):
        while self.run_flag:
            # 속도 개선 프레임 개선
            self.count += 1
            if self.count % 3 != 1:
                continue

            ret, frame_origin = self.cap.read()
            if ret is None:
                print("end of video")
                break
            frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2RGB)
            #영상 전처리기 사용
            frame = self.pre.start_processing(frame_origin)

            height, width, channel = frame.shape
            bytePerLine = channel * width
            qframe = QImage(frame.data,
                            width, height, bytePerLine,
                            QImage.Format_RGB888)
            self.changePixmap.emit(qframe)

            loop = QEventLoop()
            QTimer.singleShot(24, loop.quit)
            loop.exec_()

    def stop(self):
        self.run_flag = False
        self.quit()
        self.wait()

    def switch_flag(self):
        self.run_flag = not self.run_flag

#main class
class video_loader(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #slot 연결
        self.startButton.clicked.connect(self.start_btn_clicked)
        self.stopButton.clicked.connect(self.stop_btn_clicked)

        # 영상 쓰레드 생성
        self.th = VideoThread()
        self.th.changePixmap.connect(self.setImage)

    #화면에 영상 출력
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.video_frame.setPixmap(QPixmap.fromImage(image))

    def start_btn_clicked(self): #시작요청
        if self.th.run_flag is False:
            self.th.switch_flag()
            self.th.start()
            self.startButton.setText("Pause")
        elif self.th.run_flag is True: #일시정지 요청
            self.th.switch_flag()
            self.startButton.setText("Resume")

    def stop_btn_clicked(self):
        self.video_frame.clear()
        self.th.stop()
        self.startButton.setText("Start")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    #메인 화면 생성
    MainWindow = video_loader()
    MainWindow.show()

    sys.exit(app.exec_())