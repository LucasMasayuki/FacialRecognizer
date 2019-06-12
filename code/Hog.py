class Hog(object):
    def __init__(self):
        self.winSize = (20,20)
        self.blockSize = (10,10)
        self.blockStride = (5,5)
        self.cellSize = (10,10)
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = -1.
        self.histogramNormType = 0
        self.L2HysThreshold = 0.2
        self.gammaCorrection = 1
        self.nlevels = 64
        self.signedGradients = True
        self.useSignedGradients = 1

    def getDescriptor():
        hog = cv.HOGDescriptor(
            self.winSize,
            self.blockSize,
            self.blockStride,
            self.cellSize,
            self.nbins,
            self.derivAperture,
            self.winSigma,
            self.histogramNormType,
            self.L2HysThreshold,
            self.gammaCorrection,
            self.nlevels,
            self.useSignedGradients
        )

        return hog