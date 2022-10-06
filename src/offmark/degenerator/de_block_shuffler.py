import numpy as np
import cv2

class DeBlockShuffler:

	def __init__(self, key=None, blk_shape=(35, 30)):
		self.key = key
		self.blk_shape = blk_shape

	def set_shape(self, payload_shape):
		self.payload_shape = payload_shape
		return self

	def degenerate(self, wm, shape=(135, 240)):
		wm_shape = wm.shape
		wm = wm.astype(np.float32)
		wm = cv2.resize(wm, (shape[1], shape[0]))
		wm = self.derandomize_channel(wm, self.key, blk_shape=self.blk_shape)
		wm = cv2.resize(wm, (self.payload_shape[1], self.payload_shape[0]))
		return wm

	def derandomize_channel(self, channel, key, blk_shape=(8, 8)):
	    rows = channel.shape[0] // blk_shape[0] * blk_shape[0]
	    cols = channel.shape[1] // blk_shape[1] * blk_shape[1]
	    blks = np.array([[
	        channel[i:i + blk_shape[0], j:j + blk_shape[1]]
	        for j in range(0, cols, blk_shape[1])
	    ] for i in range(0, rows, blk_shape[0])])
	    shape = blks.shape
	    blks = blks.reshape(-1, blk_shape[0], blk_shape[1])
	    blk_num = blks.shape[0]
	    indices = np.arange(blk_num)
	    np.random.RandomState(key).shuffle(indices)
	    res = np.zeros(blks.shape)
	    res[indices] = blks
	    res = np.concatenate(np.concatenate(res.reshape(shape), 1), 1)
	    full_res = np.copy(channel)
	    full_res[:rows, :cols] = res
	    return full_res
