import math

import numpy as np


class Shuffler:

    def __init__(self, key=None):
        self.key = key

    @staticmethod
    def wm_type():
        return "bits"

    def generate_wm(self, payload, capacity):
        """Shuffles the payload and repeats it to get to capacity.
        """
        length = np.array(capacity).prod()
        payload = np.copy(payload)
        wm_len = np.array(payload.shape).prod()
        c = int(math.ceil(length / wm_len))
        np.random.RandomState(self.key).shuffle(payload)
        wm = np.stack([payload for _ in range(c)], axis=0).flatten()[:length]
        wm = wm.reshape(capacity)
        return wm
