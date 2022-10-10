import math
import warnings

import numpy as np


class GrayScale:

    def __init__(self, key=None):
        self.key = key

    @staticmethod
    def wm_type():
        return "grayscale"

    def generate_wm(self, payload, capacity):
        """Converts the watermark image pixels to 0 or 1.
        """
        size = np.array(capacity).prod()
        wm_len = np.array(payload.shape).prod()

        if wm_len > size:
            warnings.warn(
                "\nImage size {0} is greater than the embed's capacity: {1} pixels".format(payload.shape, size),
                stacklevel=3)

        payload = (payload > 127).astype(np.uint8).flatten()
        c = int(math.ceil(size / wm_len))
        np.random.RandomState(self.key).shuffle(payload)
        wm = np.stack([payload for _ in range(c)], axis=0).flatten()[:size]
        return wm.reshape(capacity)
