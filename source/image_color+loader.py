import cv2
import numpy as np

hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

def color_detec():
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3

    if hsv < 10:
        #print("case1")
        lower_blue1 = np.array([hsv-10+180, 30, 30])
        upper_blue1 = np.array([180, 255, 255])
        lower_blue2 = np.array([0, 30, 30])
        upper_blue2 = np.array([hsv, 255, 255])
        lower_blue3 = np.array([hsv, 30, 30])
        upper_blue3 = np.array([hsv+10, 255, 255])
        #     print(i-10+180, 180, 0, i)
        #     print(i, i+10)
    elif hsv > 170:
        #print("case2")
        lower_blue1 = np.array([hsv, 30, 30])
        upper_blue1 = np.array([180, 255, 255])
        lower_blue2 = np.array([0, 30, 30])
        upper_blue2 = np.array([hsv+10-180, 255, 255])
        lower_blue3 = np.array([hsv-10, 30, 30])
        upper_blue3 = np.array([hsv, 255, 255])
        #     print(i, 180, 0, i+10-180)
        #     print(i-10, i)
    else:
        #print("case3")
        lower_blue1 = np.array([hsv, 30, 30])
        upper_blue1 = np.array([hsv+10, 255, 255])
        lower_blue2 = np.array([hsv-10, 30, 30])
        upper_blue2 = np.array([hsv, 255, 255])
        lower_blue3 = np.array([hsv-10, 30, 30])
        upper_blue3 = np.array([hsv, 255, 255])
        #     print(i, i+10)
        #     print(i-10, i)
    #print(hsv)
    #print("@1", lower_blue1, "~", upper_blue1)
    #print("@2", lower_blue2, "~", upper_blue2)
    #print("@3", lower_blue3, "~", upper_blue3)

cv2.namedWindow('img_color')

#파일 불러오기
capture = cv2.VideoCapture('../../videos/4K Drone Football Footage_cut.mp4')
count = 0

#모든 프레임 재생
while (capture.isOpened()):
    has_frame, frame = capture.read()
    count = count +1
    if not has_frame:
        print('Reached the end of the video')
        break

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_detec()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv2.bitwise_and(frame, frame, mask=img_mask)


    if count % 10 == 0 :
        #원본화면 출력
        cv2.imshow('main_frame', frame)
        cv2.imshow('img_mask', img_mask)
        cv2.imshow('img_result', img_result)
        count = 0

    #키입력 대기
    #키값이 낮을수록 영상이 빠름
    key = cv2.waitKey(3)
    if key == 27:
        print('pressed ESC')
        break

cv2.destroyAllWindows()

