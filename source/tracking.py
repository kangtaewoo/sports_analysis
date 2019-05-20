import cv2
from imutils.video import VideoStream
import imutils

#재생 부분
cap = cv2.VideoCapture("../../videos/4K Drone Football Footage_cut.mp4")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))


# 트래커 초기화
tracker = cv2.TrackerCSRT_create()
# _, first_frame = cap.read()
# first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# first_gray = cv2.GaussianBlur(first_gray, (5,5), 0)

initBB = None

#재생 부분
while True:
    ret, frame = cap.read()

    if frame is None:
        break
        
    frame = imutils.resize(frame, width=600)

    if initBB is not None:
        (success, box) = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame',frame)

    key = cv2.waitKey(3) & 0xff
    
    if key == ord("s"):
        
        initBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
