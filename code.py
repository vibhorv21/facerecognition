#!/usr/bin/python

import cv2, os
import numpy as np
from PIL import Image
import sys
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.createLBPHFaceRecognizer()


def get_images_and_labels():
    X=[]
    Y=[]
    overall=0
    print "Collecting Vibhor Data"
    for root, dirs, files in os.walk("./data/vibhor", topdown=False):
        for image_path in files :
            if '.jpg' in image_path :
                Y.append(1)
                link = "./data/vibhor/" + image_path
                image_pil = Image.open(link).convert('L')
                image = np.array(image_pil, 'uint8')
                X.append(image)
    print "Collecting Harsh Data"
    for root, dirs, files in os.walk("./data/harsh", topdown=False):
        for image_path in files :
            if '.jpg' in image_path :
                Y.append(2)
                link = "./data/harsh/" + image_path
                image_pil = Image.open(link).convert('L')
                image = np.array(image_pil, 'uint8')
                X.append(image)
    print "Collecting Jakhar Data"
    for root, dirs, files in os.walk("./data/jakhar", topdown=False):
        for image_path in files :
            if '.jpg' in image_path :
                Y.append(3)
                link = "./data/jakhar/" + image_path
                image_pil = Image.open(link).convert('L')
                image = np.array(image_pil, 'uint8')
                X.append(image)
    return X, Y



path="./data"
X, Y = get_images_and_labels()
recognizer.train(X, np.array(Y))

video_capture = cv2.VideoCapture(0)
i=0
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+w,x:x+w]
        cv2.imwrite("./temp" +  ".jpg", face)
        image_pil = Image.open('./temp.jpg').convert('L')
        image = np.array(image_pil, 'uint8')
        predicted= recognizer.predict(image)
        if predicted == 1 :
            cv2.putText(frame,'Vibhor',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
        elif predicted == 2 :
            cv2.putText(frame,'Harsh',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
        elif predicted == 3 :
            cv2.putText(frame,'Jakhar',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)



    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
