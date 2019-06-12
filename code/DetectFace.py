import numpy as np
import cv2 as cv
from Hog import Hog
import os
from numpy import array

class DetectFace(object):
    """docstring for DetectFace"""

    def __init__(self):
        newHog = Hog()
        mypath = os.getcwd()
        historygramArray = []
        filename = ''
        for (root, dirs, files) in os.walk(mypath):
            for name in files:
                filename = name
                print(os.path.join(root, name))
                face = self.detectFace(name)
                print(face)

                descriptor = newHog.compute(face)
                print(descriptor)
                historygramArray.append(descriptor)

        print(filename)


    @staticmethod
    def detectFace(self, filename):
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv.imshow('img',img)
        cv.waitKey(0) 
        cv.destroyAllWindows()

        return img