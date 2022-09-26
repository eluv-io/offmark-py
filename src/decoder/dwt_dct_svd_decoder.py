import numpy as np
import pywt
import cv2

class DwtDctSvdDecoder:

    def __init__(self, key=None, scales=[0,15,0], blk=4):
        self.key = key
        self.scales = scales
        self.blk = blk

    def decode(self, yuv):
        (row, col, channels) = yuv.shape
        self.block_num = row * col // 4 // (self.blk * self.blk)
        wm_bits = np.zeros(shape=(3, self.block_num))
        for channel in range(3):
            if self.scales[channel] <= 0:
                continue
            ca, hvd = pywt.dwt2(yuv[:row // 4 * 4,:col // 4 * 4, channel], 'haar')
            self.__decode_frame(ca, self.scales[channel], wm_bits[channel])
        return np.array(wm_bits[1]).reshape(1, -1)

    def __decode_frame(self, frame, scale, wm_bits):
        (row, col) = frame.shape
        c = 0
        for i in range(row // self.blk):
            for j in range(col // self.blk):
                blk = frame[i * self.blk : i * self.blk + self.blk,
                            j * self.blk : j * self.blk + self.blk]
                wm_bit = self.__blk_extract_wm(blk, scale)
                wm_bits[c] = wm_bit
                c += 1

    def __blk_extract_wm(self, blk, scale):
        u,s,v = np.linalg.svd(cv2.dct(blk))
        wm = int((s[0] % scale) > scale * 0.5)
        return wm