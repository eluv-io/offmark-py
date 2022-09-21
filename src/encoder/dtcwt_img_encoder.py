import numpy as np
import cv2
import dtcwt
from tqdm import tqdm

import time
import multiprocessing
import heapq

from .utils import rebin, randomize_channel, derandomize_channel

default_scale = 1.5

class DtcwtImgEncoder:

    def __init__(self, key=0, str=1.0, step=5.0, blk_shape=(35, 30)):
        self.key = key
        self.alpha = default_scale * str
        self.step = step
        self.blk_shape = blk_shape

    def read_wm(self, wm):
        (w, h) = self.__infer_wm_shape(frame_shape)
        wm_len = np.array(wm.shape).prod()
        c = np.ceil(w * h / wm_len)
        self.wm = np.stack([wm for _ in range(c)], axis=0).flatten().reshape(w, h)

    def wm_capacity(self, frame_shape):
        (w, h) = self.__infer_wm_shape(frame_shape)
        return w * h

    def encode(self, yuv):
        wm_transform = dtcwt.Transform2d()
        wm_coeffs = wm_transform.forward(self.wm, nlevels=1)
        yuv_transform = dtcwt.Transform2d()
        yuv_coeffs = yuv_transform.forward(yuv[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(yuv[:, :, 0], nlevels=3)

        # Masks for the level 3 subbands
        masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(rebin(masks3[i], shape3) * (1 / self.step))
            masks3[i] *= 1.0 / max(12.0, np.amax(masks3[i]))
        for i in range(6):
            coeff = wm_coeffs.highpasses[0][:, :, i]
            h, w = coeff.shape
            coeffs = np.zeros(masks3[i].shape, dtype='complex_')
            coeffs[:h, :w] = coeff
            coeffs[-h:, :w] = coeff
            coeffs[:h, -w:] = coeff
            coeffs[-h:, -w:] = coeff
            yuv_coeffs.highpasses[2][:, :, i] += self.alpha * (masks3[i] * coeffs)
        yuv[:, :, 1] = yuv_transform.inverse(yuv_coeffs)
        return yuv

    def prepare_wm(self, wm_path, img_shape):
        wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        assert wm is not None, "Watermark not found in {}".format(wm_path)
        wm_shape = self.__infer_wm_shape(img_shape)
        wm = cv2.resize(wm, (wm_shape[1], wm_shape[0]))
        self.wm = (wm > 127).astype(np.uint8) * 255
        self.wm = self.wm.astype(np.int32)
        self.wm[self.wm != 255] = -255
        self.wm = randomize_channel(self.wm, self.key, blk_shape=self.blk_shape)

    def __infer_wm_shape(self, frame_shape):
        w = (((frame_shape[0] + 1) // 2 + 1) // 2 + 1) // 2
        h = (((frame_shape[1] + 1) // 2 + 1) // 2 + 1) // 2
        if w % 2 == 1:
            w += 1
        if h % 2 == 1:
            h += 1
        return (w, h)

class DtcwtImgDecoder:

    def __init__(self, wm_shape, key=None, str=1.0, step=5.0, blk_shape=(35, 30)):
        self.key = key
        wm_len = np.array(wm.shape).prod()
        self.wm_len = wm_len
        self.alpha = default_scale * str
        self.step = step
        self.blk_shape = blk_shape

    def decode(self, yuv):
        wmed_transform = dtcwt.Transform2d()
        wmed_coeffs = wmed_transform.forward(yuv[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(yuv[:, :, 0], nlevels=3)

        masks3 = [0 for i in range(6)]
        inv_masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(rebin(masks3[i], shape3) * (1.0 / self.step))
            masks3[i][masks3[i] == 0] = 0.01
            masks3[i] *= 1.0 / max(12.0, np.amax(masks3[i]))
            inv_masks3[i] = 1.0 / masks3[i]

        shape = wmed_coeffs.highpasses[2][:,:,i].shape
        h, w = (shape[0] + 1) // 2, (shape[1] + 1) // 2
        coeffs = np.zeros((h, w, 6), dtype='complex_')
        for i in range(6):
            coeff = (wmed_coeffs.highpasses[2][:,:,i]) * inv_masks3[i] * 1 / self.alpha
            coeffs[:,:,i] = coeff[:h, :w] + coeff[:h, -w:] + coeff[-h:, :w] + coeff[-h:, -w:]
        highpasses = tuple([coeffs])
        lowpass = np.zeros((h * 2, w * 2))
        t = dtcwt.Transform2d()
        wm = t.inverse(dtcwt.Pyramid(lowpass, highpasses))

        wm_bits = wm.flatten()
        wm_avg = np.zeros(shape=self.wm_len)
        for i in range(self.wm_len):
            wm_avg[i] = wm_bits[1, i::self.wm_len].mean()

        return wm
