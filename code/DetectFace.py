import numpy as np
from Hog import Hog
from Lbp import Lbp
import cv2
import os
from numpy import array

class DetectFace(object):
    """docstring for DetectFace"""

    def __init__(self):
        currentPath = os.path.abspath(__file__)
        mypath = os.path.abspath(os.path.join(currentPath, '..', '..', 'img'))

        hogHistorygramArray = []
        lbpHistogramArray = []

        hogDescriptor = Hog.getDescriptor()
        filename = ''
        for (root, dirs, files) in os.walk(mypath):
            for name in files:
                pathOfFile = os.path.join(mypath, '1', name)
                face = self.detectFace(pathOfFile)

                lbpComputed = Lbp.compute(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))

                hogComputed = hogDescriptor.compute(face)
                hogHistorygramArray.append(hogComputed)
                lbpHistogramArray.append(lbpComputed)

    @staticmethod
    def detectFace(pathOfFile):
        face_cascade = cv2.CascadeClassifier('C:\\Users\Lucas\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        img = cv2.imread(pathOfFile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

        print('Faces found: ', len(faces))

        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img
