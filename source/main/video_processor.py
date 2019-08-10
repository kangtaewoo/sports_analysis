import math

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSlot


class Video_Processor():
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__instance
        return cls.__instance

    def __init__(self):
        #팀 색상
        self.lower_white = np.array([0, 0, 242])
        self.upper_white = np.array([255, 255, 255])
        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

        #팀별 공격방향
        self.leftBlue = 0
        self.rightBlue = 0
        self.centerBlue = 0
        self.leftWhite = 0
        self.centerWhite = 0
        self.rightWhite = 0
        self.resLB = 0
        self.resCB = 0
        self.resRB = 0
        self.resLW = 0
        self.resCW = 0
        self.resRW = 0

        self.TeamW = np.zeros((400, 400, 3), np.uint8) + 255
        self.TeamB = np.zeros((400, 400, 3), np.uint8) + 255

        #배경제거 알고리즘
        self.fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        self.kernelBig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.fgmask = None

        #트래커
        self.initBB = None
        self.pts3 = [(0, 0)]
        self.tracker = cv2.TrackerCSRT_create()
        self.ret = None
        self.distance = 0

    def start_processing(self, image):
        self.frame = image

        self.frame = self.angle_corrector(self.frame)
        self.fgmask = self.background_subtractor(self.frame)
        self.draw_boundary(self.fgmask)
        self.set_tracker(self.initBB)
        self.update_tracker()
        self.calculate()
        graph1, graph2 = self.draw_graph()

        return self.frame, graph1, graph2

    #각도 보정
    def angle_corrector(self, frame):
        # cut field , warp 좌표 순서 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝 (포인트 수동지정)
        pts1 = np.float32([(0, 110), (1280, 161), (0, 585), (1280, 585)])
        pts2 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
        m = cv2.getPerspectiveTransform(pts1, pts2)
        frame = cv2.warpPerspective(frame, m, (1280, 720))
        return frame

    #배경제거
    def background_subtractor(self, frame):
        fgmask = self.fgbg.apply(frame)
        fgmask = self.noise_compensator(fgmask)
        return fgmask

    #노이즈 보정
    def noise_compensator(self, fg_image):
        # fg_image = cv2.morphologyEx(fg_image, cv2.MORPH_OPEN, self.kernel)
        # fg_image = cv2.morphologyEx(fg_image, cv2.MORPH_CLOSE, self.kernel)

        fg_image = cv2.erode(fg_image, self.kernel, iterations=1)
        fg_image = cv2.dilate(fg_image, self.kernel, iterations=1)
        fg_image = cv2.dilate(fg_image, self.kernelBig, iterations=1)
        fg_image = cv2.erode(fg_image, self.kernelBig, iterations=1)
        return fg_image

    def draw_boundary(self, fgmask):
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = self.filtercontours(contours)
        dict = self.classifycontours(contours)
        for c in dict['ateam']:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(self.frame, (x, y), (x + 10, y + 30), (255, 255, 255), 2)
            if y < 132:
                self.leftWhite += 1
            elif y < 451:
                self.centerWhite += 1
            elif y > 451:
                self.rightWhite += 1

        for c in dict['bteam']:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(self.frame, (x, y), (x + 10, y + 30), (0, 0, 255), 2)
            if y < 132:
                self.rightBlue += 1
            elif y < 451:
                self.centerBlue += 1
            else:
                self.leftBlue += 1

        # 선수 윤곽선 검출
    def filtercontours(self, contours):
        playercontours = list()
        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            # if y < 670 and y > 30 and w > 5 and h > 5 and h >= 1.3*w and h < 40 and x > 90 and x < 1200:
            if w * h > 70 and y < 600:
                playercontours.append(c)
        return playercontours

        # 선수 팀 판별
    def classifycontours(self, contours):
        classfiedObjects = {}
        ateamplayers = list()
        bteamplayers = list()

        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            crop_img = self.frame[y:y + h, x:x + w]
            meanColor = cv2.mean(crop_img)
            if meanColor[1] > 145:
                ateamplayers.append(c)
            else:
                bteamplayers.append(c)
        classfiedObjects['ateam'] = ateamplayers
        classfiedObjects['bteam'] = bteamplayers
        return classfiedObjects

    def calculate(self):
        sumWhite = self.leftWhite + self.centerWhite + self.rightWhite
        sumBlue = self.leftBlue + self.centerBlue + self.rightBlue

        if sumWhite != 0:
            self.resLW = (self.leftWhite / sumWhite) * 100
            self.resCW = (self.centerWhite / sumWhite) * 100
            self.resRW = (self.rightWhite / sumWhite) * 100
        else:
            self.resLW = 0
            self.resCW = 0
            self.resRW = 0

        if sumBlue != 0:
            self.resLB = (self.leftBlue / sumBlue) * 100
            self.resCB = (self.centerBlue / sumBlue) * 100
            self.resRB = (self.rightBlue / sumBlue) * 100
        else:
            self.resLB = 0
            self.resCB = 0
            self.resRB = 0

        info1 = [
            ("Team", "White"),
            ("Right", "{:.2f}".format(self.resRW)),
            ("Center", "{:.2f}".format(self.resCW)),
            ("Left", "{:.2f}".format(self.resLW)),
        ]

        # for (i, (k, v)) in enumerate(info1):
        #     text = "{}: {}".format(k, v)
        #     cv2.putText(frame, text, (W - 400, H - ((i * 20) + 20)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info2 = [
            ("Team", "Blue"),
            ("Left", "{:.2f}".format(self.resLB)),
            ("Center", "{:.2f}".format(self.resCB)),
            ("Right", "{:.2f}".format(self.resRB)),
        ]

        # for (i, (k, v)) in enumerate(info2):
        #     text = "{}: {}".format(k, v)
        #     cv2.putText(frame, text, (W - 200, H - ((i * 20) + 20)),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def draw_graph(self):
        # Team White Attack direction Rate
        self.TeamW = cv2.rectangle(self.TeamW, (70, int(300 - self.resRW)), (130, 300), (255, 0, 0), -1)
        self.TeamW = cv2.putText(self.TeamW, "Right", (75, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        self.TeamW = cv2.putText(self.TeamW, "{0}%".format(str(int(self.resRW))), (77, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)

        self.TeamW = cv2.rectangle(self.TeamW, (170, int(300 - self.resCW)), (230, 300), (0, 255, 0), -1)
        self.TeamW = cv2.putText(self.TeamW, "Center", (171, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.TeamW = cv2.putText(self.TeamW, "{0}%".format(str(int(self.resCW))), (173, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        self.TeamW = cv2.rectangle(self.TeamW, (270, int(300 - self.resLW)), (330, 300), (0, 0, 255), -1)
        self.TeamW = cv2.putText(self.TeamW, "Left", (278, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        self.TeamW = cv2.putText(self.TeamW, "{0}%".format(str(int(self.resLW))), (280, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

        # Team Blue Attack direction Rate
        self.TeamB = cv2.rectangle(self.TeamB, (70, int(300-self.resRB)), (130, 300), (255,0,0), -1)
        self.TeamB = cv2.putText(self.TeamB, "Right", (75, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        self.TeamB = cv2.putText(self.TeamB, "{0}%".format(str(int(self.resRB))), (77, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        self.TeamB = cv2.rectangle(self.TeamB, (170, int(300-self.resCB)), (230, 300), (0,255,0), -1)
        self.TeamB = cv2.putText(self.TeamB, "Center", (171, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.TeamB = cv2.putText(self.TeamB, "{0}%".format(str(int(self.resCB))), (173, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.TeamB = cv2.rectangle(self.TeamB, (270, int(300-self.resLB)), (330, 300), (0,0,255), -1)
        self.TeamB = cv2.putText(self.TeamB, "Left", (278, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        self.TeamB = cv2.putText(self.TeamB, "{0}%".format(str(int(self.resLB))), (280, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return self.TeamW, self.TeamB

    #선수 추적(tracking)
    # @pyqtSlot(tuple)
    def set_tracker(self, rect):
        self.initBB = rect
        if self.initBB is not None:
            self.tracker.init(self.frame, self.initBB)

    def update_tracker(self):
        if self.initBB is not None:
            (success, box) = self.tracker.update(self.frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                # (prex, prey) = self.pts3.pop()
                # if prex != 0:
                #     diff = math.sqrt(math.pow((x * 0.01 - prex * 0.01), 2) + math.pow((y * 0.16 - prey * 0.16), 2))
                #     if diff < 0.1:
                #         diff = 0
                #     self.distance += diff
                # print(self.distance)
                # self.pts3.append((x, y))
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (225, 0, 0), 2)

#for testing
if __name__ == "__main__":
    cap = cv2.VideoCapture("../../videos/ test1.mp4")

    pre = Video_Processor()

    #재생 부분
    while True:
        ret, frame_origin = cap.read()
        if frame_origin is None:
            break
        # frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2RGB)
        processed_frame = pre.start_processing(frame_origin)

        cv2.imshow('preprocessed_frame', processed_frame)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

    # cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
