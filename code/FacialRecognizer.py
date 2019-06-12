import numpy

class FacialRecognizer(object):
    """docstring for FacialRecognizer"""

    def __init__(self, arg):
        super(FacialRecognizer, self).__init__()
        self.arg = arg

        x = numpy.array([0, 1, 0, 1], [0, 0, 1, 1])
        yd = numpy.array([1, 0, 0, 1])
        h = 1
        neural_network(x, yd, h)
