import cv2 as cv
class Hog(object):
    @staticmethod
    def getDescriptor():
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

        return cv.HOGDescriptor(
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
            nlevels
        )
