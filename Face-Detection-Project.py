# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:24:13 2018

@author: Tamer
"""

# Importing the libraries
import cv2

# Loading the cascades
# Load the cascade for the face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load the cascade for the eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defaine a function that will do the detection
def detect(gray, frame):
    # apply the detect detectMultiScale from face_cascade object to locate
    # one or several faces in the image
    # gray because cascade work on black and white umages
    # 1.3 Scale factor which tell the size of image will be reduced means
    # how much filter 'kernel' will be increased which 1.3 times by experiment
    # 5 Minimum number of neighbours that will be accepted by expirment
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # For each face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # Detect the eyes in refrential of faces to speed computation
        # Get the zone of detected rectangle for balck/white image and colored image
        grayRec = gray[y:y+h, x:x+w]
        frameRec = frame[y:y+h, x:x+w]
        # apply the detectMultiScale from eye_cascade object to locate 
        # one or several eyes in the image
        eyes = eye_cascade.detectMultiScale(grayRec, 1.1, 3)
        # Fom each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eyes with refrential of face.
            cv2.rectangle(frameRec, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
    return frame

# Doing some face recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detectedimg = detect(gray, frame)
    cv2.imshow('Video', detectedimg)
    if cv2.waitKey(1) == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()












