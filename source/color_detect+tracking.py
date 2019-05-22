import cv2
from imutils.video import VideoStream
import imutils
import numpy as np

def filtercontours(contours):
    playercontours = list()
    for c in contours:
        rect = cv2.boundingRect(c)
        if(rect[2] < 7 or rect[3] < 20) or (rect[2] > 60 or rect[3] > 100): continue
        playercontours.append(c)
    return playercontours

#재생 부분
cap = cv2.VideoCapture("../../videos/test.mp4")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))

# fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)
# fgbg1 = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)

lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

lower_red = np.array([160,60,50])
upper_red = np.array([280,255,255])

lower_white = np.array([0, 0, 0])
upper_white = np.array([0, 0, 255])

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

idx = 0
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
        
    frame = imutils.resize(frame, width=1000)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((5,5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        if(h>=w):
            if(w<25 and h < 25) and (x > 239 or y > 128) and (x < 892 or y > 104):
                idx = idx+1
                player_img = frame[y:y+h,x:x+w]
                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(player_hsv, lower_red, upper_red)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                nzCountred = cv2.countNonZero(res1)
                mask2 = cv2.inRange(player_hsv, lower_white, upper_white)
                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res2)
                # mask3 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                # res3 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                # res3 = cv2.cvtColor(res3,cv2.COLOR_HSV2BGR)
                # res3 = cv2.cvtColor(res3,cv2.COLOR_BGR2GRAY)
                # nzCountblue = cv2.countNonZero(res3)

                if(nzCountred >= 1):
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                else:
                     pass

                if(nzCount >= 1):
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(255,255,255),2)
                else:
                     pass

                # if(nzCountblue >= 1):
                #     cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
                # else:
                #      pass     

    # cv2.bitwise_not(mask, mask)
    # fgmask = fgbg.apply(frame)

    # # 노이즈 제거
    # fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    # fgmask = cv2.erode(fgmask, kernel, iterations=1)

    # clonedFrame = fgmask
    # ret, thresh1 = cv2.threshold(clonedFrame, 100, 255, cv2.THRESH_BINARY)

    # fgmask = cv2.bitwise_and(fgmask, mask)
    # contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filtercontours(contours)
    # for c in contours:
    #     rect = cv2.boundingRect(c)
    #     x, y, w, h = rect
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255),2)



    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    # for index, centroid in enumerate(centroids):
    #     if stats[index][0] == 0 and stats[index][1] == 0:
    #         continue
    #     if np.any(np.isnan(centroid)):
    #         continue


    #     x, y, width, height, area = stats[index]
    #     centerX, centerY = int(centroid[0]), int(centroid[1])

        # if area > 100:
        #     cv2.circle(frame, (centerX, centerY), 1, (0, 0, 255), 2)
        #     cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
        # print(centerX, centerY)
    


    if initBB is not None:
        (success, box) = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame',frame)
    # cv2.imshow('fg', fgmask)
    key = cv2.waitKey(3) & 0xff
    
    if key == ord("s"):
        
        initBB = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
