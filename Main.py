import cv2 as cv
import numpy as np
import face_recognition
import pandas

cap = cv.VideoCapture(1)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv.imshow('Frame', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()