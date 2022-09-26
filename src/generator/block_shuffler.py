import cv2
import numpy as np    

class BlockShuffler:
    def __init__(self, key=None):
        self.key = key
        self.blk_shape = (8, 8)

    def wm_type(self):
        return "grayscale"

    def generate_wm(self, payload, capacity):
        wm = cv2.resize(payload, (capacity[1], capacity[0]))
        payload = (payload > 127).astype(np.uint8)
        payload = payload.astype(np.int32)
        payload[payload != 255] = -255
        payload = self.randomize_channel(payload, self.key, blk_shape=self.blk_shape)
        payload = payload.flatten()
        wm_len = np.array(payload.shape).prod()
        c = int(math.ceil(capacity / wm_len))
        wm = np.stack([payload for _ in range(c)], axis=0).flatten()[:capacity]
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
