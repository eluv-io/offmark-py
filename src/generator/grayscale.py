import numpy as np
import warnings
import math

class GrayScale:
	
	def __init__(self, key=None):
		self.key = key

	def wm_type(self):
		return "grayscale"

	def generate_wm(self, payload, capacity):
		size = np.array(capacity).prod()
		wm_len = np.array(payload.shape).prod()

		if wm_len > size:
			warnings.warn("\nImage size {0} is greater than the encoder's capacity: {1} pixels".format(payload.shape, size), stacklevel=3)

		payload = (payload > 127).astype(np.uint8).flatten()
		c = int(math.ceil(size / wm_len))
		np.random.RandomState(self.key).shuffle(payload)
		wm = np.stack([payload for _ in range(c)], axis=0).flatten()[:size]
		return wm.reshape(capacity)
