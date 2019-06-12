import numpy as np
from Hog import Hog
import cv2 as cv
import os
from numpy import array

class DetectFace(object):
    """docstring for DetectFace"""

    def __init__(self):
        mypath ='/home/lucas/FacialRecognizer/img/1'
        hogHistorygramArray = []
        lbpHistogramArray = []
        filename = ''
        for (root, dirs, files) in os.walk(mypath):
            print('passa')
            for name in files:
                pathOfFile = mypath + "/" + name
                face = self.detectFace(pathOfFile)

                descriptor = Hog.getDescriptor()
                computed = descriptor.compute(face)
                hogHistorygramArray.append(computed)

        print(hogHistorygramArray)
        print(lbpHistogramArray)

    @staticmethod
    def detectFace(pathOfFile):
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
        img = cv.imread(pathOfFile)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        return img
