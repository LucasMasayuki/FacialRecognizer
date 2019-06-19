import numpy as np
from Hog import Hog
from Lbp import Lbp
import cv2
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
            for name in files:
                pathOfFile = mypath + "/" + name
                face = self.detectFace(pathOfFile)

                # #hogDescriptor = Hog.getDescriptor()
                # lbpDescriptor = Lbp.getDescriptor(face)
                # #hogComputed = hogDescriptor.compute(face)
                # lbpComputed = lbpDescriptor.compute(face)
                # #hogHistorygramArray.append(hogComputed)
                # lbpHistogramArray.append(lbpComputed)

        # print(hogHistorygramArray)
        # print(lbpHistogramArray)

    @staticmethod
    def detectFace(pathOfFile):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        print cv2.__version__

        img = cv2.imread(pathOfFile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

        print('Faces found: ', len(faces))

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_color)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew,ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img
