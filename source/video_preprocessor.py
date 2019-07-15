import queue

import cv2
import numpy as np


class Video_Preprocessor():

    lower_white = np.array([0, 0, 242])
    upper_white = np.array([255, 255, 255])

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    q = queue.Queue(10)
    point = list()
    idx = 0

    def start_processing(self,image):
        self.frame = image
        self.angle_corrector()
        self.background_subtractor()
        self.labeling()
        return self.frame

    #각도 보정
    def angle_corrector(self):
        # cut field , warp 좌표 순서 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝 (포인트 수동지정)
        pts1 = np.float32([(0, 110), (1280, 161), (0, 585), (1280, 585)])
        pts2 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
        m = cv2.getPerspectiveTransform(pts1, pts2)
        self.frame = cv2.warpPerspective(self.frame, m, (1280, 720))

    #배경제거
    def background_subtractor(self):
        # res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        # res_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)
        self.fgmask = self.fgbg.apply(self.frame)
        self.fgmask = self.noise_compensator(self.fgmask)

    #노이즈 보정
    def noise_compensator(self, fg_image):
        fg_image = cv2.morphologyEx(fg_image, cv2.MORPH_OPEN, self.kernel)
        fg_image = cv2.morphologyEx(fg_image, cv2.MORPH_CLOSE, self.kernel)
        return fg_image

    def labeling(self):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.fgmask)
        for index, centroid in enumerate(centroids):
            if stats[index][0] == 0 and stats[index][1] == 0:
                continue
            if np.any(np.isnan(centroid)):
                continue

            x, y, w, h, area = stats[index]
            # centerX, centerY = int(centroid[0]), int(centroid[1])

            if h >= 1.5*w and h > 15 and h < 30 and x > 90 and y < 690:
                self.point.append([int(x+w/2), int(y+h/2)])
                if len(self.point) > 100 :
                     del self.point[0]
                # count = 20
                for i in self.point :
                    # count -= 1
                    self.q.put_nowait(i)
                    if self.q.qsize() == 10 :
                        temp = self.q.get_nowait()
                        cv2.circle(self.frame,(temp[0],temp[1]),1,(255,255,255),3)
                    # cv2.circle(frame,((q.get())[0]),((q.get())[1]),1,(255,255,255),2)
                    # cv2.circle(frame, (centerX, centerY), 1, (0, 0, 255), 2)
                    # cv2.rectangle(frame, (x, y), (x+10, y+30), (255, 0, 0), 2)
                self.idx = self.idx+1
                player_img = self.frame[y:y+h,x:x+w]
                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(player_hsv, self.lower_blue, self.upper_blue)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                nzCountblue = cv2.countNonZero(res1)
                mask2 = cv2.inRange(player_hsv, self.lower_white, self.upper_white)
                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res2)

                if(nzCountblue >= 1):
                    cv2.rectangle(self.frame, (x,y),(x+w,y+h),(255,0,0),2)
                    # if x < 615:
                    #     if y < 132:
                    #         rightBlue += 1
                    #     elif y < 451:
                    #         centerBlue += 1
                    #     else:
                    #         leftBlue += 1
                else:
                    pass

                if(nzCount >= 20):
                    cv2.rectangle(self.frame, (x,y),(x+10,y+30),(255,255,255),2)
                    # if x > 615:
                    #     if y < 132:
                    #         leftWhite += 1
                    #     elif y < 451:
                    #         centerWhite += 1
                    #     elif y > 451:
                    #         rightWhite += 1
                else:
                    pass


if __name__ == "__main__":
    cap = cv2.VideoCapture("../../videos/test1.mp4")

    #허프변환
    # ret, first_frame = cap.read()
    # cv2.imwrite("../../videos/test1.png", first_frame)
    # img = cv2.imread("../../videos/test1.png")
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(imgray, 40, 170)
    # lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=160)
    #
    # for line in lines:
    #     r, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*r
    #     y0 = b*r
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0+1000*(a))
    #     x2 = int(x0-1000*(-b))
    #     y2 = int(y0-1000*(a))
    #
    #     cv2.line(img, (x1,y1), (x2, y2), (255,0,0), 1)
    #
    # cv2.imshow('edges',img)

    # for min in range(0, 200, 10):
    #     for max in range(0, 200, 10):
    #         edges = cv2.Canny(imgray, min, max)
    #         cv2.imwrite("../../videos/Canny/canny%d_%d.png" %(min,max), edges)


    pre = Video_Preprocessor()

    #재생 부분
    while True:
        ret, frame_origin = cap.read()
        if frame_origin is None:
            break
        # frame_origin = cv2.cvtColor(frame_origin, cv2.COLOR_BGR2RGB)
        preprocessed_frame = pre.start_processing(frame_origin)

        cv2.imshow('preprocessed_frame', preprocessed_frame)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

    # cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
