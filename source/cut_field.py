import cv2
import numpy as np

#파일 불러오기
capture = cv2.VideoCapture('../../videos/test.mp4')
# #subtractor 정의
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=70)

# #모양 바꿀 수 있음
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


#모든 프레임 재생
while (capture.isOpened()):
    ret,frame = capture.read()
    if not ret:
        print('Reached the end of the video')
        break

    #cut field , warp 좌표 순서 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝 (포인트 수동지정)
    pts1 = np.float32([(90,80), (765,97), (5,393), (860,394)])
    pts2 = np.float32([[0,0], [640,0], [0,480], [640,480]])
    m = cv2.getPerspectiveTransform(pts1,pts2)
    frame = cv2.warpPerspective(frame,m,(640,480))

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 배경제거
    mask_player = fgbg.apply(frame)
    # 노이즈 제거
    mask_player = cv2.morphologyEx(mask_player, cv2.MORPH_OPEN, kernel)

    lower_green = np.array([36, 0, 0])
    upper_green = np.array([86, 255, 255])
    # Threshold the HSV image to get only green colors
    mask_field = cv2.inRange(img_hsv, lower_green, upper_green)
    # cv2.bitwise_not(mask_field, mask_field)


    frame_player = cv2.bitwise_and(frame, frame, mask=mask_player)
    frame_field = cv2.bitwise_and(frame, frame, mask=mask_field)
    cv2.imshow('frame_player', frame_player)
    cv2.imshow('frame_field', frame_field)
    cv2.imshow('mask_field', mask_field)

    #키입력 대기 (키값이 낮을수록 영상이 빠름)
    key = cv2.waitKey(1)

    if key == 27:
        print('pressed ESC')
        break

cv2.destroyAllWindows()