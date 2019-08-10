import sys

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
    send_player = pyqtSignal(tuple)

    def __init__(self, video_path):
        super().__init__()
        self.running = False
        self.mt_pause = False
        self.mt_stop = False
        self.mutex = QMutex()
        self.mt_pause_condition = QWaitCondition()

        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
        self.pre = Video_Processor.instance()
        self.initBB = None
        # self.count = 0
        # self.count_end = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def __del__(self):
        self.cap.release()
        del self.pre

    def pause(self):
        print(sys.stderr, "pause btn clicked")
        self.mt_pause = True

    def resume(self):
        print(sys.stderr, "resume btn clicked")
        self.mt_pause = False
        self.mt_pause_condition.wakeAll()

    def stop(self):
        print(sys.stderr, "stop btn clicked")
        self.mt_stop = True
        # self.wait()
        # self.quit()

    def reset(self):
        self.mt_stop = False
        self.mt_pause = False
        self.mutex.unlock()
        self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
        self.sec = 0
        self.sec_changed.emit(self.sec)
        self.pre.initBB = None
        self.wait()
        print(sys.stderr, "video thread reset")

    def run(self):
        print(sys.stderr, "start btn clicked")
        print(sys.stderr, "video thread start")
        self.running = True
        while self.running:
            QThread.msleep(10)
            self.mutex.lock()
            if self.mt_stop is True:
                break
            if self.mt_pause is True:
                print(sys.stderr, "video thread pause")
                self.mt_pause_condition.wait(self.mutex)
                print(sys.stderr, "video thread resume")
            self.mutex.unlock()
            print(sys.stderr, "video thread is running")
            # 속도 개선 프레임 개선
            ret, frame_origin = self.cap.read()
            if ret is False:
                print("end of video")
                # self.stop()
                break
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) % 3 != 1:
                continue

            frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2RGB)
            #영상 전처리기 사용
            self.frame, graph1, graph2 = self.pre.start_processing(frame_origin)
            #영상 qframe 형식 변환
            frame_origin = self.convert(frame_origin)
            self.frame = self.convert(self.frame)
            graph1 = self.convert(graph1)
            graph2 = self.convert(graph2)

            self.changePixmap_origin.emit(frame_origin)
            self.changePixmap_processed.emit(self.frame)
            self.changePixmap_graph1.emit(graph1)
            self.changePixmap_graph2.emit(graph2)
            self.sec_changed.emit((self.cap.get(cv2.CAP_PROP_POS_MSEC))/1000)

            loop = QEventLoop()
            QTimer.singleShot(10, loop.quit)
            loop.exec_()
        print(sys.stderr, "video thread stopped")
        self.reset()

    def convert(self, img):
        height, width, channel = img.shape
        bytePerLine = channel * width
        qframe = QImage(img.data,
                        width, height, bytePerLine,
                        QImage.Format_RGB888)
        return qframe


    def switch_flag(self):
        self.mt_pause = not self.mt_pause

    def select_player(self, rect):
        print("player selected")
        # self.initBB = rect
        self.pre.initBB = rect
        # if rect is not None:
        #     # self.initBB = cv2.selectROI("Form", self.frame, fromCenter=False, showCrosshair=True)
        #     # self.initBB = cv2.selectROI(self.frame, fromCenter=False)
        #     ret = self.pre.set_tracker(rect)
        #     print(ret)

    @pyqtSlot(int)
    def time_change(self, time):
        self.sec = time
        # self.cap.set(cv2.CAP_PROP_POS_MSEC, time)
