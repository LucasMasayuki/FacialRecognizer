from random import random
import math
class Mlp(object):
    def __init__(
        self,
        X,
        Yd,
        X,
        h,
        N
    ):
        self.h = h
        self.ne = ne
        self.ns = ns

        self.network = list()

        self.max_times = 50

        self.alfa = 1

        hiddenLayer = [[random() for i in range(ne + 1)] for i in range(h)]
        self.network.append(hiddenLayer)
        outputLayer = [[random() for i in range(h + 1)] for i in range(ns)]
        self.network.append(outputLayer)

    def train() {

    }

    def feedForward()
        zin = [[0 for x in range(N)] for y in range(h)]
        z = [[0 for x in range(N)] for y in range(h + 1)]
        yin = [[0 for x in range(N)] for y in range(h)]
        y = [[0 for x in range(N)] for y in range(h)]

        At = A.T

        zin = numpy.dot(X, At)

        z = numpy.divide(1, math.exp(-zin) + 1)
        z = [z, [1 for x in range(N)]]
        Bt = B.T
        yin = numpy.dot(z, Bt)
        y = numpy.divide(1, math.exp(-yin) + 1)

        return y