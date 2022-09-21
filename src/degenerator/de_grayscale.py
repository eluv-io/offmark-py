import numpy as np

class DeGrayScale:

	def __init__(self, key=None):
		self.key = key

	def set_shape(self, payload_shape):
		self.payload_shape = payload_shape
		self.payload_len = np.array(payload_shape).prod()
		self.payload_idx = np.arange(self.payload_len)
		np.random.RandomState(self.key).shuffle(self.payload_idx)
		return self

	def degenerate(self, wm_bits):
		payload = np.zeros(shape=self.payload_len)
		for i in range(self.payload_len):
			payload[i] = wm_bits[i::self.payload_len].mean()
		payload[self.payload_idx] = payload.copy()
		res = (payload > 0.5).astype(np.uint8) * 255
		return res.reshape(self.payload_shape)
