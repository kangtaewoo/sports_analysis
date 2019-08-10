import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from source.main.tread import VideoThread
from source.ui.video_loader_dialog import Ui_Form


class VideoLoader(QWidget, Ui_Form):

    time_change_signal = pyqtSignal(int)
    release = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self.setupUi(self)

        # 쓰레드
        self.th_video = VideoThread(video_path)

        #slot 연결
        self.startButton.clicked.connect(self.start_btn_clicked)
        self.stopButton.clicked.connect(self.stop_btn_clicked)

        self.th_video.changePixmap_origin.connect(self.setOrigin)
        self.th_video.changePixmap_processed.connect(self.setProcessed)
        self.th_video.changePixmap_graph1.connect(self.setGraph1)
        self.th_video.changePixmap_graph2.connect(self.setGraph2)

        # self.horizontalSlider.sliderMoved.connect(self.value_change)
        self.th_video.sec_changed.connect(self.time_update)
        self.time_change_signal.connect(self.th_video.time_change)


    def time_convert(self, msec):
        sec = int(msec/1000)
        return sec

    #화면에 영상 출력
    @pyqtSlot(QImage)
    def setOrigin(self, image):
        self.video_frame1.clear()
        self.video_frame1.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setProcessed(self, image):
        self.video_frame2.clear()
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

    # #재생바 옮겼을때 값 변화
    # def value_change(self):
    #     self.time = self.horizontalSlider.value()
    #     self.time_change_signal.emit(self.time)

    def start_btn_clicked(self):
        btn_text = self.startButton.text()
        if btn_text == "Start":
            self.th_video.start()
            self.video_frame2.flag = False
            self.startButton.setText("Pause")
        elif btn_text == "Pause":
            self.th_video.pause()
            self.video_frame2.flag = True
            self.startButton.setText("Resume")
        elif btn_text == "Resume":
            self.video_frame2.rubberband.hide()
            print(self.video_frame2.rect)
            self.th_video.select_player(self.video_frame2.rect)
            self.th_video.resume()
            self.video_frame2.flag = False
            self.startButton.setText("Pause")

    def stop_btn_clicked(self):
        self.th_video.stop()
        self.video_frame1.clear()
        self.video_frame2.clear()
        self.team_a_graph.clear()
        self.team_b_graph.clear()
        self.startButton.setText("Start")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    #메인 화면 생성
    video_loader = VideoLoader("../../../videos/test1.mp4")
    video_loader.show()

    sys.exit(app.exec_())