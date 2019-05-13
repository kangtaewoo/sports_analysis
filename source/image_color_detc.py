import cv2
import numpy as np

img_color = cv2.imread('1.jpg')

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

Bmask = cv2.inRange(img_hsv, np.array([100, 80, 80]), np.array([140, 255, 255])) #각각의 범위 지정
Gmask = cv2.inRange(img_hsv, np.array([35, 80, 80]), np.array([80, 255, 255]))
Rmask = cv2.inRange(img_hsv, np.array([-10, 100, 100]), np.array([30, 255, 255]))

B = cv2.bitwise_and(img_color, img_color, mask=Bmask)
G = cv2.bitwise_and(img_color, img_color, mask=Gmask)
R = cv2.bitwise_and(img_color, img_color, mask=Rmask)

cv2.imshow('defalut', img_color)
cv2.imshow('blue', B)
cv2.imshow('green', G)
cv2.imshow('red', R)

cv2.waitKey(0)
cv2.destroyAllWindows()