import cv2
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from source.main.tread import VideoThread
from source.ui.video_loader_dialog import Ui_Form


class VideoLoader(QWidget, Ui_Form):

    time_change_signal = pyqtSignal(float)

    def __init__(self, video_path):
        super().__init__()
        self.setupUi(self)

        self.cap = cv2.VideoCapture(video_path)

        #slot 연결
        self.startButton.clicked.connect(self.start_btn_clicked)
        self.stopButton.clicked.connect(self.stop_btn_clicked)
        # self.horizontalSlider.valueChanged.connect(self.value_change)
        # self.horizontalSlider.sliderPressed.connect(self.value_change)
        self.horizontalSlider.sliderMoved.connect(self.value_change)

        # 영상 쓰레드 생성
        self.th_video = VideoThread(self.cap)
        self.th_video.changePixmap_origin.connect(self.setOrigin)
        self.th_video.changePixmap_processed.connect(self.setProcessed)
        self.th_video.changePixmap_graph1.connect(self.setGraph1)
        self.th_video.changePixmap_graph2.connect(self.setGraph2)
        self.th_video.sec_changed.connect(self.time_update)
        self.time_change_signal.connect(self.th_video.time_change)

        # timer 초기화
        timerVar = QTimer()


    #화면에 영상 출력
    @pyqtSlot(QImage)
    def setOrigin(self, image):
        self.video_frame1.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setProcessed(self, image):
        self.video_frame2.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setGraph1(self, image):
        self.team_a_graph.clear()
        self.team_a_graph.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setGraph2(self, image):
        self.team_b_graph.clear()
        self.team_b_graph.setPixmap(QPixmap.fromImage(image))

    #재생시간 출력
    @pyqtSlot(float)
    def time_update(self, sec):
        self.time = sec
        self.cur_time.clear()
        self.cur_time.setText("time:%.2f" % (self.time))
        self.horizontalSlider.setValue(self.time)

    #재생바 옮겼을때 값 변화
    def value_change(self):
        self.time = self.horizontalSlider.value()
        self.time_change_signal.emit(self.time)


    def start_btn_clicked(self): #시작요청
        if self.th_video.running is True:
            self.th_video.switch_flag()
            self.startButton.setText("Resume")
        else :
            self.th_video.switch_flag()
            self.th_video.start()
            self.startButton.setText("Pause")



    def stop_btn_clicked(self):
        self.video_frame1.clear()
        self.video_frame2.clear()
        self.th_video.stop()
        self.startButton.setText("Start")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    #메인 화면 생성
    video_loader = VideoLoader("../../videos/test1.mp4")
    video_loader.show()

    sys.exit(app.exec_())