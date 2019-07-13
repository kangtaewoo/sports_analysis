import cv2
import numpy as np
import time
import math
#재생 부분
cap = cv2.VideoCapture("../../videos/test1.mp4")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
kernelBig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

tracker = cv2.TrackerCSRT_create()
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50)

initBB = None
count = 0
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
# _, first_frame = cap.read()
# first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# first_gray = cv2.GaussianBlur(first_gray, (5,5), 0)

def filtercontours(contours):
    playercontours = list()
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        # if y < 670 and y > 30 and w > 5 and h > 5 and h >= 1.3*w and h < 40 and x > 90 and x < 1200:
        if w * h > 70 and y < 600:
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
        if meanColor[1] > 145:
            ateamplayers.append(c)
        else:
            bteamplayers.append(c)
    classfiedObjects['ateam'] = ateamplayers
    classfiedObjects['bteam'] = bteamplayers
    return classfiedObjects

#재생 부분
while True:

    start = time.time()
    ret, frame_origin = cap.read()

    # 속도 개선 프레임 개선
    count+=1
    if count%3 != 1 :
        continue

    pts1 = np.float32([(0,110), (1280,161), (0,585), (1280,585)])
    pts2 = np.float32([[0,0], [1280,0], [0,720], [1280,720]])
    m = cv2.getPerspectiveTransform(pts1,pts2)
    frame = cv2.warpPerspective(frame_origin,m,(1280,720))
    (H, W) = frame.shape[:2]

    fgmask = fgbg.apply(frame)

    # 노이즈 제거
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    fgmask = cv2.dilate(fgmask, kernelBig, iterations=1)
    fgmask = cv2.erode(fgmask, kernelBig, iterations=1)

    clonedFrame = fgmask

    ret, thresh1 = cv2.threshold(clonedFrame, 100, 255, cv2.THRESH_BINARY)
    

    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    # for index, centroid in enumerate(centroids):
    #     if stats[index][0] == 0 and stats[index][1] == 0:
    #         continue
    #     if np.any(np.isnan(centroid)):
    #         continue


    #     x, y, width, height, area = stats[index]
    #     centerX, centerY = int(centroid[0]), int(centroid[1])

    #     # if area > 50 and height > 1.2*width and height < 40 and width > 7:
    #     # cv2.circle(frame, (centerX, centerY), 1, (0, 0, 255), 2)
    #     if area > 50 and y < 600:
    #         cv2.rectangle(frame, (x, y), (x + 10, y + 30), (255, 0, 0), 2)


    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # refcontours, hierarchy = cv2.findContours(reffgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = filtercontours(contours)
    # refcontours = filtercontours(refcontours)
    dict = classifycontours(contours)

    for c in dict['ateam']:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + 10, y + 30), (255, 255, 255), 2)
        if y < 132:  
            leftWhite += 1
        elif y < 451:
            centerWhite += 1
        elif y > 451:
            rightWhite += 1


    for c in dict['bteam']:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + 10, y + 30), (255, 0, 0), 2)
        if y < 132:
            rightBlue += 1
        elif y < 451:
            centerBlue += 1
        else:
            leftBlue += 1

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
            ("Distance", "{:.2f}".format(distance)),
    ]
   
    if distance != 0:
        for (i, (k, v)) in enumerate(info3):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('mask',fgmask)
    cv2.imshow('frame',frame)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

    if key == ord("s"):
        
        initBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)

    end = time.time()
    seconds = end - start

cap.release()
cv2.destroyAllWindows()
