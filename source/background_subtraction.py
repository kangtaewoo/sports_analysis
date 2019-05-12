import cv2
import numpy as np
#재생 부분
cap = cv2.VideoCapture("../../videos/4K Drone Football Footage_cut.mp4")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=70)

# _, first_frame = cap.read()
# first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# first_gray = cv2.GaussianBlur(first_gray, (5,5), 0)

#재생 부분
while True:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    # 노이즈 제거
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue


        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            cv2.circle(frame, (centerX, centerY), 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)


    cv2.imshow('mask',fgmask)
    cv2.imshow('frame',frame)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
