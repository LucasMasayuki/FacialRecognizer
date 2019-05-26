import numpy as np
import cv2 as cv
from os import walk

class DetectFace(object):
    """docstring for DetectFace"""

    def __init__(self, arg):
        super(DetectFace, self).__init__()
        self.arg = arg
        f = []
        hog = self.hog()
        for (dirpath, dirnames, filenames) in walk(mypath):
            face = self.detectFace(filenames)
            break

    def detectFace(filename):
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

    def hog():
        winSize = (20,20)
        blockSize = (10,10)
        blockStride = (5,5)
        cellSize = (10,10)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradients = True

        hog = cv.HOGDescriptor(
            winSize,
            blockSize,
            blockStride,
            cellSize,
            nbins,
            derivAperture,
            winSigma,
            histogramNormType,
            L2HysThreshold,
            gammaCorrection,
            nlevels,
            useSignedGradients
        )

        return hog