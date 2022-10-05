import numpy as np
import cv2

class CorrShuffler:
	
	def __init__(self, key=None):
		self.key = key

	def wm_type(self):
		return "bits"

	def generate_wm(self, payload, capacity, shape=(1080, 1920)):
		# payload is unnecessary
		wm = np.random.RandomState(self.key).randint(0, 2, shape).astype(np.float32)
		wm[wm == 0] = -1
		wm = cv2.resize(wm, (capacity[1], capacity[0]))
		return wm
