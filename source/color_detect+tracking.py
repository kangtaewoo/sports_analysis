import cv2
from imutils.video import VideoStream
import imutils
import numpy as np
import math
import time

#재생 부분
cap = cv2.VideoCapture("../../videos/test1.mp4")
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))


fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)
# fgbg1 = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)

lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

lower_red = np.array([160,60,50])
upper_red = np.array([280,255,255])

lower_white = np.array([0, 0, 242])
upper_white = np.array([255, 255, 255])

lower_blue = np.array([100,50,50])
upper_blue = np.array([130,255,255])

YELLOW_MIN = np.array([45, 50, 50], np.uint8)
YELLOW_MAX = np.array([80, 100, 100], np.uint8)

idx = 0
# 트래커 초기화
tracker = cv2.TrackerCSRT_create()
# tracker = cv2.TrackerMedianFlow_create()
# _, first_frame = cap.read()
# first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# first_gray = cv2.GaussianBlur(first_gray, (5,5), 0)

initBB = None
count=0
pts3 = [(0, 0)]
distance = 0
resLW = 0
resCW = 0
resRW = 0
resLB = 0
resCB = 0
resRB = 0

leftBlue = 0
rightBlue = 0
centerBlue = 0

leftWhite = 0
centerWhite = 0
rightWhite = 0
#재생 부분
while True:
    start = time.time()
    ret, frame_origin = cap.read()
    if frame_origin is None:
        break

    # 속도 개선 프레임 개선
    # count+=1
    # if count%2 != 1 :
    #     continue

    #cut field , warp 좌표 순서 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝 (포인트 수동지정)
    pts1 = np.float32([(0,110), (1280,161), (0,585), (1280,585)])
    pts2 = np.float32([[0,0], [1280,0], [0,720], [1280,720]])
    m = cv2.getPerspectiveTransform(pts1,pts2)
    frame = cv2.warpPerspective(frame_origin,m,(1280,720))
    (H, W) = frame.shape[:2]
    # frame_origin = imutils.resize(frame_origin, width=720)
    
    # frame_player = cv2.bitwise_and(frame, frame, mask=mask_player)
    # frame_field = cv2.bitwise_and(frame, frame, mask=mask_field)
    # cv2.imshow('frame_player', frame_player)
    # cv2.imshow('frame_field', frame_field)
    # cv2.imshow('mask_field', mask_field)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # Histogram Equalization
    # h, s, v = cv2.split(hsv)
    # equalizedV = cv2.equalizeHist(v)
    # hsv = cv2.merge([h,s,equalizedV])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # refMask = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
    # cv2.bitwise_not(refMask, refMask)
    # refMask = cv2.erode(refMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=1)
    # refMask = cv2.dilate(refMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=1)


    # kernel = np.ones((5,5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    fgmask = fgbg.apply(frame)

    # 노이즈 제거
    # fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    # fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    # reffgMask = cv2.bitwise_and(fgmask, refMask)
    # notrefmask = cv2.bitwise_not(reffgMask)
    # fgmask = cv2.bitwise_and(fgmask, notrefmask)
     # 속도 개선 프레임 개선
    count+=1
    if count%3 != 1 :
        continue

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue


        x, y, w, h, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        # if h >= 1.5*w and h > 15 and h < 30 and x > 90:
        #     cv2.circle(frame, (centerX, centerY), 1, (0, 0, 255), 2)
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if h >= 1.5*w and h > 15 and h < 30 and x > 90 and y < 690 and (y + h/2) < 690:

                    # cv2.circle(frame, (centerX, centerY), 1, (0, 0, 255), 2)
                    # cv2.rectangle(frame, (x, y), (x+10, y+30), (255, 0, 0), 2)
                idx = idx+1
                player_img = frame[y:y+h,x:x+w]
                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                nzCountblue = cv2.countNonZero(res1)
                mask2 = cv2.inRange(player_hsv, lower_white, upper_white)
                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res2)

                if(nzCountblue >= 1):
                    cv2.rectangle(frame, (x,y),(x+10,y+30),(255,0,0),2)
                    if x < 615:
                        if y < 132:
                            rightBlue += 1
                        elif y < 451:
                            centerBlue += 1
                        else:
                            leftBlue += 1
                else:
                    pass

                if(nzCount >= 20):
                    cv2.rectangle(frame, (x,y),(x+10,y+30),(255,255,255),2)
                    if x > 615:
                        if y < 132:
                            leftWhite += 1
                        elif y < 451:
                            centerWhite += 1
                        elif y > 451:
                            rightWhite += 1
                else:
                    pass

                sumWhite = leftWhite+centerWhite+rightWhite
                sumBlue = leftBlue+centerBlue+rightBlue

                if sumWhite != 0:
                    resLW = (leftWhite/sumWhite) * 100
                    resCW = (centerWhite/sumWhite) * 100
                    resRW = (rightWhite/sumWhite) * 100
                else :
                    resLW = resCW = resRW = 0
                if sumBlue != 0:
                    resLB = (leftBlue/sumBlue) * 100
                    resCB = (centerBlue/sumBlue) * 100
                    resRB = (rightBlue/sumBlue) * 100
                else :
                    resLB = 0
                    resCB = 0
                    resRB = 0

    info1 = [
            ("Team", "White"),
            ("Right", "{:.2f}".format(resRW)),
            ("Center", "{:.2f}".format(resCW)),
            ("Left", "{:.2f}".format(resLW)),
    ]

    for (i, (k, v)) in enumerate(info1):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (W - 400, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    info2 = [
            ("Team", "Blue"),
            ("Left", "{:.2f}".format(resLB)),
            ("Center", "{:.2f}".format(resCB)),
            ("Right", "{:.2f}".format(resRB)),
   
    ]

    for (i, (k, v)) in enumerate(info2):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (W - 200, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

          

    # cv2.bitwise_not(mask, mask)
    

    # clonedFrame = fgmask
    # ret, thresh1 = cv2.threshold(clonedFrame, 100, 255, cv2.THRESH_BINARY)

    # fgmask = cv2.bitwise_and(fgmask, mask)
    # contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filtercontours(contours)
    # for c in contours:
    #     rect = cv2.boundingRect(c)
    #     x, y, w, h = rect
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255),2)



    
    # frame = imutils.resize(frame, width=800)
    # frame_origin = imutils.resize(frame_origin, width=800)


    if initBB is not None:
        (success, box) = tracker.update(frame)

        if success:
            
            (x, y, w, h) = [int(v) for v in box]
            (prex, prey) = pts3.pop()
            if prex != 0:
                diff = math.sqrt(math.pow((x*0.01 - prex*0.01), 2) + math.pow((y*0.16 - prey*0.16), 2))
                if diff < 0.1 :
                    diff = 0
                distance += diff
            print(distance)
            pts3.append((x, y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)

    info3 = [
            ("Name", "Player1"),
            ("Distance", "{:.2f}m".format(distance)),
    ]
   
    if distance != 0:
        for (i, (k, v)) in enumerate(info3):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # frame_origin = cv2.resize(frame_origin, dsize=(720,480), interpolation=cv2.INTER_LINEAR)

    # frame_origin = cv2.resize(frame_origin, dsize=(0,0),fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    # fgmask = cv2.resize(fgmask, dsize=(0,0),fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    # # 전시용 모니터 사이즈full
    # frame = cv2.resize(frame, dsize=(0,0),fx=1.5, fy=1.4, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('frame_origin',frame_origin)
    cv2.imshow('fg', fgmask)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xff
    
    if key == ord("s"):
        
        initBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)

    if key == 27:
        break

    end = time.time()
    seconds = end - start
    print("seconds : {0}".format(seconds))

cap.release()
cv2.destroyAllWindows()
