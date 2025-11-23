import cv2
import numpy as np
import time
import requests
from datetime import datetime

#カメラ起動
cap = cv2.VideoCapture(0)
time.sleep(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

def detect_motion(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    __, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    cv2.imshow("dilated", dilated)
    cv2.waitKey(1)
    contours, __ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #小さいノイズは無視
    for c in contours:
        if cv2.contourArea(c) > 500:
            return True

    return False

while True:
    if detect_motion(frame1, frame2):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}.jpg"
        cv2.imwrite(filename, frame1)

    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret:
        break

time.sleep(0.03)
