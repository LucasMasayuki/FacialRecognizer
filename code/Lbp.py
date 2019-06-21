from skimage import feature
import numpy as np

# code from https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
class Lbp(object):
    @staticmethod
    def compute(image, eps = 1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(
            image,
            24,
            8
        )

        (hist, _) = np.histogram(
            lbp.ravel(),
            bins = np.arange(0, 24 + 3),
            range=(0, 24 + 2)
        )

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist
