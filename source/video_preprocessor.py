import cv2
import numpy as np

class Video_Preprocessor():

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def start_processing(self,image):
        self.frame = image
        self.angle_corrector()
        self.background_subtractor()
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
        #필드 색상 검출(초록색 검출 잘안됨)
        # hsv = cv2.cvtColor(self.frame, cv2.COLOR_RGB2HSV)
        # mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        # res = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        # res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        # res_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)
        self.fgmask = self.fgbg.apply(self.frame)
        self.fgmask = self.noise_compensator(self.fgmask)

    #노이즈 보정
    def noise_compensator(self, fg_image):
        fg_image = cv2.morphologyEx(fg_image, cv2.MORPH_OPEN, self.kernel)
        fg_image = cv2.morphologyEx(fg_image, cv2.MORPH_CLOSE, self.kernel)
        return fg_image


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
