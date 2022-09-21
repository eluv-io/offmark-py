import numpy as np
import math

class GrayScale:
	
	def __init__(self, key=None):
		self.key = key

	def wm_type(self):
		return "grayscale"

	def generate_wm(self, payload, capacity):
		payload = (payload > 127).astype(np.uint8).flatten()
		wm_len = np.array(payload.shape).prod()
		c = int(math.ceil(capacity / wm_len))
		np.random.RandomState(self.key).shuffle(payload)
		wm = np.stack([payload for _ in range(c)], axis=0).flatten()[:capacity]
		return wm
