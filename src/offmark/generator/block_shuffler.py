import cv2
import numpy as np


class BlockShuffler:
    def __init__(self, key=None, blk_shape=(35, 30)):
        self.key = key
        self.blk_shape = blk_shape

    @staticmethod
    def wm_type():
        return "grayscale"

    def generate_wm(self, payload, capacity, shape=(135, 240)):
        """Scrambles the watermark image by shuffling around blocks of pixels.
        """
        wm = cv2.resize(payload, (shape[1], shape[0]))
        wm = self.randomize_channel(wm, self.key, blk_shape=self.blk_shape)
        wm = cv2.resize(wm, (capacity[1], capacity[0]))
        wm = (wm > 127).astype(np.uint8) * 255
        wm = wm.astype(np.int32)
        wm[wm != 255] = -255
        return wm

    def randomize_channel(self, channel, key, blk_shape=(8, 8)):
        rows = channel.shape[0] // blk_shape[0] * blk_shape[0]
        cols = channel.shape[1] // blk_shape[1] * blk_shape[1]
        blks = np.array([[
            channel[i:i + blk_shape[0], j:j + blk_shape[1]]
            for j in range(0, cols, blk_shape[1])
        ] for i in range(0, rows, blk_shape[0])])
        shape = blks.shape
        blks = blks.reshape(-1, blk_shape[0], blk_shape[1])
        np.random.RandomState(key).shuffle(blks)
        full_res = np.copy(channel)
        res = np.concatenate(np.concatenate(blks.reshape(shape), 1), 1)
        full_res[:rows, :cols] = res
        return full_res
