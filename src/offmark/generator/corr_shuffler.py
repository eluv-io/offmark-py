import cv2
import numpy as np


class CorrShuffler:

    def __init__(self, key=None):
        self.key = key

    @staticmethod
    def wm_type():
        return "bits"

    def generate_wm(self, payload, capacity, shape=(1080, 1920)):
        """Generate a 2-D array with elements of either 1 or -1. Though the
        values may change after scaling due to resampling/interpolating

        >>> c = CorrShuffler(key=0)
        >>> c.generate_wm(None, (3, 3))
        array([[ 0.5, -0.5,  1. ],
               [ 0.5,  0.5,  0. ],
               [ 0.5,  0.5, -0.5]], dtype=float32)
        """
        # payload is unnecessary
        wm = np.random.RandomState(self.key).randint(0, 2, shape).astype(np.float32)
        wm[wm == 0] = -1
        wm = cv2.resize(wm, (capacity[1], capacity[0]))
        return wm
