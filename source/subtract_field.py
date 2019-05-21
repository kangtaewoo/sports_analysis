import cv2
import numpy as np

#파일 불러오기
capture = cv2.VideoCapture('../../cutvideo.mp4')
# count = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=70)

hsv = 0
red1 = range(0,20,1)
red2 = range(170,180,1)
yellow = range(20,40, 1)
green = range(40,80,1)
skyblue = range(80,110,1)
blue = range(110,140,1)
pink = range(140,170,1)

lower_color = 0
upper_color = 0

def mouse_callback(event, x, y, flags, param):
    global hsv, lower_color, upper_color, red1, red2, yellow, green, skyblue, blue, pink
    #좌클릭시 위치에 있는 픽셀값을 읽어와서 HSV로 변환한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        color = frame[y, x]
        one_pixel = np.uint8([[color]])
        print('bgr:', one_pixel)

        hsv = cv2.cvtColor(one_pixel, cv2.COLOR_BGR2HSV)
        print('hsv:',hsv)
        hsv = hsv[0][0]

        #hsv[0] 값은 bgr2hsv 컨버팅 한 값 이므로 0~180의 범위를 갖는다.
        #red : 0~10, 170~180
        if hsv[0] in red1:
            print("case red")
            lower_color = np.array([hsv[0],30,30])
            upper_color = np.array([hsv[0],255,255])
        elif hsv[0] in red2:
            print("case red")
            lower_color = np.array([hsv[0],30,30])
            upper_color = np.array([hsv[0],255,255])
        elif hsv[0] in yellow:
            print("case yellow")
            lower_color = np.array([hsv[0],30,30])
            upper_color = np.array([hsv[0],255,255])
        elif hsv[0] in green:
            print("case green")
            lower_color = np.array([hsv[0],30,30])
            upper_color = np.array([hsv[0],255,255])
        elif hsv[0] in skyblue:
            print("case red")
            lower_color = np.array([hsv[0],30,30])
            upper_color = np.array([hsv[0],255,255])
        elif hsv[0] in blue:
            print("case blue")
            lower_color = np.array([hsv[0],30,30])
            upper_color = np.array([hsv[0],255,255])
        elif hsv[0] in pink:
            print("case red")
            lower_color = np.array([hsv[0],30,30])
            upper_color = np.array([hsv[0],255,255])

        print(hsv[0])
        print(lower_color, "~", upper_color)


cv2.namedWindow('main_frame')
cv2.setMouseCallback('main_frame', mouse_callback)

#모든 프레임 재생
while (capture.isOpened()):
    has_frame, frame = capture.read()
    # count = count +1
    if not has_frame:
        print('Reached the end of the video')
        break

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    fgmask = fgbg.apply(frame)

    # 노이즈 제거
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # # 필드 삭제
    # # define range of green color in HSV
    # lower_green = np.array([36, 0, 0])
    # upper_green = np.array([86, 255, 255])
    # # Threshold the HSV image to get only green colors
    # img_mask_field = cv2.inRange(img_hsv, lower_green, upper_green)
    # cv2.bitwise_not(img_mask_field, img_mask_field)

    img_mask1 = cv2.inRange(img_hsv, lower_color, upper_color)
    # img_mask = img_mask1
    fg_mask = img_mask1
    # img_mask_field | img_mask1

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    # result_frame = cv2.bitwise_and(frame, frame, mask=img_mask)
    fg_frame = cv2.bitwise_and(fgmask, fgmask, mask = fg_mask)

    # if count % 10 == 0 :
    #화면 출력
    # cv2.imshow('img_color', img_color)

    frame = cv2.resize(frame, (640, 480))
    # img_mask = cv2.resize(img_mask, (640, 480))
    # result_frame = cv2.resize(result_frame, (640, 480))
    fg_frame = cv2.resize(fg_frame, (640, 480))

    cv2.imshow('main_frame', frame)
    # cv2.imshow('img_mask', img_mask)
    # cv2.imshow('result_frame', result_frame)
    cv2.imshow('fg_frame', fg_frame)
    # count = 0
    #키입력 대기
    #키값이 낮을수록 영상이 빠름
    key = cv2.waitKey(1)
    if key == 27:
        print('pressed ESC')
        break

cv2.destroyAllWindows()