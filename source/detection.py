# 사람 detection
import cv2
from matplotlib import pyplot as plt

body_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_fullbody.xml')
imageFile = './test.jpg'
img = cv2.imread(imageFile)
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bodies = body_cascade.detectMultiScale(grayImage, scaleFactor=1.2, maxSize=(100,130))
for (x,y,w,h) in bodies:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0) , 2)

plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()