import cv2

#파일 불러오기
capture = cv2.VideoCapture('../videos/4K Drone Football Footage_cut.mp4')
count = 0

#모든 프레임 재생
while (capture.isOpened()):
    has_frame, frame = capture.read()
    count = count +1
    if not has_frame:
        print('Reached the end of the video')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if count % 10 == 0 :
        #원본화면 출력
        cv2.imshow('main_frame', frame)
        cv2.imshow('gray', gray)
        cv2.imshow('bgr2hsv1', hsv1)
        count = 0

    #키입력 대기
    #키값이 낮을수록 영상이 빠름
    key = cv2.waitKey(3)
    if key == 27:
        print('pressed ESC')
        break

cv2.destroyAllWindows()