import numpy as np
import pywt
import cv2

class DwtDctSvdEncoder:
    def __init__(self, key=None, scales=[0,15,0], blk=4):
        self.key = key
        self.scales = scales
        self.blk = blk

    def read_wm(self, wm):
        self.wm = wm[0]

    def wm_capacity(self, frame_shape):
        row, col, channels = frame_shape
        block_num = row * col // 64
        return (1, block_num)

    def encode(self, yuv):
        (row, col, channels) = yuv.shape
        for channel in range(3):
            if self.scales[channel] <= 0:
                continue
            ca, hvd = pywt.dwt2(yuv[:row // 4 * 4,:col // 4 * 4, channel], 'haar')
            self.__encode_frame(ca, self.scales[channel])
            yuv[:row // 4 * 4, :col // 4 * 4, channel] = pywt.idwt2((ca, hvd), 'haar')
        return yuv

    def __encode_frame(self, frame, scale):
        (row, col) = frame.shape
        c = 0
        for i in range(row // self.blk):
            for j in range(col // self.blk):
                blk = frame[i * self.blk : i * self.blk + self.blk,
                              j * self.blk : j * self.blk + self.blk]
                wm_bit = self.wm[c]
                embedded_blk = self.__blk_embed_wm(blk, wm_bit, scale)
                frame[i * self.blk : i * self.blk + self.blk,
                      j * self.blk : j * self.blk + self.blk] = embedded_blk
                c += 1

    def __blk_embed_wm(self, blk, wm_bit, scale):
        u, s, v = np.linalg.svd(cv2.dct(blk))
        s[0] = (s[0] // scale + 0.25 + 0.5 * wm_bit) * scale
        return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))
