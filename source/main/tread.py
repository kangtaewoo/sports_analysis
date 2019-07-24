import cv2

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from source.main.video_processor import Video_Processor


class VideoThread(QThread):

    changePixmap_origin = pyqtSignal(QImage)
    changePixmap_processed = pyqtSignal(QImage)
    changePixmap_graph1 = pyqtSignal(QImage)
    changePixmap_graph2 = pyqtSignal(QImage)

    sec_changed = pyqtSignal(float)

    def __init__(self, file):
        super().__init__()
        self.running = False
        self.cap = file
        self.pre = Video_Processor()
        # self.count = 0
        self.count_end = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def __del__(self):
        self.cap.release()

    def run(self):
        while self.running:
            # 속도 개선 프레임 개선
            ret, frame_origin = self.cap.read()
            if ret is None:
                print("end of video")
                break
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) % 3 != 1:
                continue

            frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2RGB)
            #영상 전처리기 사용
            frame, graph1, graph2 = self.pre.start_processing(frame_origin)
            #영상 qframe 형식 변환
            frame = self.convert(frame)
            frame_origin = self.convert(frame_origin)
            graph1 = self.convert(graph1)
            graph2 = self.convert(graph2)

            self.changePixmap_origin.emit(frame_origin)
            self.changePixmap_processed.emit(frame)
            self.changePixmap_graph1.emit(graph1)
            self.changePixmap_graph2.emit(graph2)
            self.sec_changed.emit((self.cap.get(cv2.CAP_PROP_POS_MSEC))/1000)

            loop = QEventLoop()
            QTimer.singleShot(30, loop.quit)
            loop.exec_()

    def convert(self, img):
        height, width, channel = img.shape
        bytePerLine = channel * width
        qframe = QImage(img.data,
                        width, height, bytePerLine,
                        QImage.Format_RGB888)
        return qframe

    def stop(self):
        self.running = False
        self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
        self.sec = 0
        self.sec_changed.emit(self.sec)
        self.wait()
        # self.quit()

    def switch_flag(self):
        self.running = not self.running

    @pyqtSlot(float)
    def time_change(self, time):
        self.sec = time
        # self.cap.set(cv2.CAP_PROP_POS_MSEC, time)
