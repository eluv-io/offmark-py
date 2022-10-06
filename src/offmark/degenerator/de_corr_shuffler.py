import numpy as np
import cv2
from scipy.signal import correlate2d

class DeCorrShuffler:

	def __init__(self, key=None):
		self.key = key

	def set_shape(self, payload_shape):
		return self

	def degenerate(self, wm, mode="fast", shape=(1080, 1920)):
		wmk = np.random.RandomState(self.key).randint(0, 2, shape).astype(np.float32)
		wmk[wmk == 0] = -1
		wmk = cv2.resize(wmk, (wm.shape[1], wm.shape[0]))
		shape = wm.shape[0] * wm.shape[1]
		if mode == "fast":
			nwm = (wm - np.mean(wm)) / np.std(wm)
			nwmk = (wmk - np.mean(wmk)) / np.std(wmk)
			corr = np.sum(nwm * nwmk) / shape
		elif mode == "slow":
			c = correlate2d(wm, wmk) / shape
			idx = np.unravel_index(c.argmax(), c.shape)
			corr = c[idx]
		print("Correlation: ", corr)
		if corr > 0.1:
			return True
		else:
			return False
