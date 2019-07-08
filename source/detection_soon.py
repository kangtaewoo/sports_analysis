import cv2
from imutils.video import VideoStream
import imutils
import numpy as np
import math
import time
import queue

#재생 부분
cap = cv2.VideoCapture("../../videos/test1.mp4")
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))


fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg1 = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)

lower_green = np.array([36, 0, 0])
upper_green = np.array([86, 255, 255])

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
point = list()
q=queue.Queue(10)

def filtercontours(contours):
    playercontours = list()
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        if w < 20 and h < 30 and y < 670 and y > 30 and h > 4:
            playercontours.append(c)
    return playercontours

def classifycontours(contours):
    classfiedObjects = {}
    ateamplayers = list()
    bteamplayers = list()

    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        crop_img = frame[y:y+h,x:x+w]
        meanColor = cv2.mean(crop_img)
        if meanColor[1] > 148:
            ateamplayers.append(c)
        else:
            bteamplayers.append(c)
    classfiedObjects['ateam'] = ateamplayers
    classfiedObjects['bteam'] = bteamplayers
    return classfiedObjects

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
kerenlBig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))

# ret, testFrame = cap.read()
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# height, width, layer = testFrame.shape
# new_h = int(height / 2)
# new_w = int(width / 2)

# frame = cv2.resize(testFrame, (new_w, new_h))
# out = cv2.VideoWriter('segmentation.avi', fourcc, 20.0, (new_w, new_h))

fgbg = cv2.createBackgroundSubtractorMOG2()


#재생 부분
while True:
    start = time.time()
    ret, frame = cap.read()
    if frame is None:
        break

    # 속도 개선 프레임 개선
    count+=1
    if count%3 != 1 :
        continue

    #cut field , warp 좌표 순서 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝 (포인트 수동지정)
    pts1 = np.float32([(0,110), (1280,161), (0,585), (1280,585)])
    pts2 = np.float32([[0,0], [1280,0], [0,720], [1280,720]])
    m = cv2.getPerspectiveTransform(pts1,pts2)
    frame = cv2.warpPerspective(frame,m,(1280,720))
    (H, W) = frame.shape[:2]

    # height, width, layers = bigFrame.shape
    # new_h = int(height / 2)
    # new_w = int(width / 2)

    # frame = cv2.resize(bigFrame, (new_w, new_h))
    # bigFrame=frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.bitwise_not(mask, mask)

    fgmask = fgbg.apply(frame)

    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)

    clonedFrame = fgmask

    ret, thresh1 = cv2.threshold(clonedFrame, 100, 255, cv2.THRESH_BINARY)
    fgmask = cv2.bitwise_and(fgmask, mask)

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = filtercontours(contours)
    dict = classifycontours(contours)

    for c in dict['ateam']:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + 10, y + 30), (255, 255, 255), 2)

    for c in dict['bteam']:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + 10, y + 30), (255, 0, 0), 2)
    # fgmask = fgbg.apply(frame)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
   

    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    # for index, centroid in enumerate(centroids):
    #     if stats[index][0] == 0 and stats[index][1] == 0:
    #         continue
    #     if np.any(np.isnan(centroid)):
    #         continue

    #     x, y, w, h, area = stats[index]
    #     centerX, centerY = int(centroid[0]), int(centroid[1])
        # if h >= 1.5*w and h > 15 and h < 30 and x > 90:
        #     cv2.circle(frame, (centerX, centerY), 1, (0, 0, 255), 2)
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # if h >= 1.5*w and h > 15 and h < 30 and x > 90 and y < 690:
        #     point.append([int(x+w/2), int(y+h/2)])
        #     if len(point) > 100 :
        #         del point[0]
        #     # count = 20
        #     for i in point :
        #         # count -= 1
        #         q.put_nowait(i)
        #         if q.qsize() == 10 :
        #             temp = q.get_nowait()
        #             cv2.circle(frame,(temp[0],temp[1]),1,(255,255,255),3)
    #         idx = idx+1
    #         player_img = frame[y:y+h,x:x+w]
    #         player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
    #         mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
    #         res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
    #         res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
    #         res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
    #         nzCountblue = cv2.countNonZero(res1)
    #         mask2 = cv2.inRange(player_hsv, lower_white, upper_white)
    #         res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
    #         res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
    #         res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
    #         nzCount = cv2.countNonZero(res2)

    #         if(nzCountblue >= 1):
    #             cv2.rectangle(frame, (x,y),(x+10,y+30),(255,0,0),2)
    #             if x < 615:
    #                 if y < 132:
    #                     rightBlue += 1
    #                 elif y < 451:
    #                     centerBlue += 1
    #                 else:
    #                     leftBlue += 1
    #         else:
    #             pass

    #         if(nzCount >= 20):
    #             cv2.rectangle(frame, (x,y),(x+10,y+30),(255,255,255),2)
    #             if x > 615:
    #                 if y < 132:  
    #                     leftWhite += 1
    #                 elif y < 451:
    #                     centerWhite += 1
    #                 elif y > 451:
    #                     rightWhite += 1
    #         else:
    #             pass

    #         sumWhite = leftWhite+centerWhite+rightWhite
    #         sumBlue = leftBlue+centerBlue+rightBlue

    #         if sumWhite != 0:
    #             resLW = (leftWhite/sumWhite) * 100
    #             resCW = (centerWhite/sumWhite) * 100
    #             resRW = (rightWhite/sumWhite) * 100
    #         else :
    #             resLW = resCW = resRW = 0
    #         if sumBlue != 0:
    #             resLB = (leftBlue/sumBlue) * 100
    #             resCB = (centerBlue/sumBlue) * 100
    #             resRB = (rightBlue/sumBlue) * 100
    #         else :
    #             resLB = 0
    #             resCB = 0
    #             resRB = 0

    # info1 = [
    #         ("Team", "White"),
    #         ("Right", "{:.2f}".format(resRW)),
    #         ("Center", "{:.2f}".format(resCW)),
    #         ("Left", "{:.2f}".format(resLW)),
    # ]

    # for (i, (k, v)) in enumerate(info1):
    #     text = "{}: {}".format(k, v)
    #     cv2.putText(frame, text, (W - 400, H - ((i * 20) + 20)),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # info2 = [
    #         ("Team", "Blue"),
    #         ("Left", "{:.2f}".format(resLB)),
    #         ("Center", "{:.2f}".format(resCB)),
    #         ("Right", "{:.2f}".format(resRB)),
    # ]

    # for (i, (k, v)) in enumerate(info2):
    #     text = "{}: {}".format(k, v)
    #     cv2.putText(frame, text, (W - 200, H - ((i * 20) + 20)),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # if initBB is not None:
    #     (success, box) = tracker.update(frame)

    #     if success:
    #         (x, y, w, h) = [int(v) for v in box]
    #         (prex, prey) = pts3.pop()
    #         if prex != 0:
    #             diff = math.sqrt(math.pow((x*0.01 - prex*0.01), 2) + math.pow((y*0.16 - prey*0.16), 2))
    #             if diff < 0.1 :
    #                 diff = 0
    #             distance += diff
    #         print(distance)
    #         pts3.append((x, y))
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)

    # info3 = [
    #         ("Name", "Player1"),
    #         ("Distance", "{:.2f}".format(distance)),
    # ]
   
    # if distance != 0:
    #     for (i, (k, v)) in enumerate(info3):
    #             text = "{}: {}".format(k, v)
    #             cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # # cv2.imshow('frame_origin',frame_origin)
    cv2.imshow('fg', fgmask)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xff
    
    # if key == ord("s"):
        
    #     initBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
    #     tracker.init(frame, initBB)

    if key == 27:
        break

    end = time.time()
    seconds = end - start
    #print("seconds : {0}".format(seconds))

cap.release()
cv2.destroyAllWindows()
