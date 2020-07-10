import cv2 as cv
import numpy as np

cap=cv.VideoCapture(0)

prev_frame=None

while True:
    _,frame=cap.read()

    frame=np.flip(frame,axis=1)
    if prev_frame is None:
        prev_frame=frame
        continue

    hsv = cv.cvtColor(frame.copy(), cv.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv, lower_red, upper_red)
    mask1+= mask2
    mask1 = cv.morphologyEx(mask1, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv.morphologyEx(mask1, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))
    mask2 = cv.bitwise_not(mask1)
    res1 = cv.bitwise_and(frame, frame, mask=mask2)
    res2 = cv.bitwise_and(prev_frame, prev_frame, mask=mask1)
    final_output = cv.addWeighted(res1, 1, res2, 1, 0)
    cv.imshow("magic", final_output)
    if cv.waitKey(1)==27:
        break

