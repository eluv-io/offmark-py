import numpy as np
import cv2
import dtcwt

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
        wm = wm.reshape(w, h)
        wm_transform = dtcwt.Transform2d()
        wm_coeffs = wm_transform.forward(wm, nlevels=1)
        self.wm_coeffs = wm_coeffs

    def wm_capacity(self, frame_shape):
        (w, h) = self.__infer_wm_shape(frame_shape)
        return w * h

    def encode(self, yuv):
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
            coeff = self.wm_coeffs.highpasses[0][:, :, i]
            h, w = coeff.shape
            coeffs = np.zeros(masks3[i].shape, dtype='complex_')
            coeffs[:h, :w] = coeff
            coeffs[-h:, :w] = coeff
            coeffs[:h, -w:] = coeff
            coeffs[-h:, -w:] = coeff
            yuv_coeffs.highpasses[2][:, :, i] += self.alpha * (masks3[i] * coeffs)
        yuv[:, :, 1] = yuv_transform.inverse(yuv_coeffs)
        return yuv

    def __infer_wm_shape(self, frame_shape):
        w = (((frame_shape[0] + 1) // 2 + 1) // 2 + 1) // 2
        h = (((frame_shape[1] + 1) // 2 + 1) // 2 + 1) // 2
        if w % 2 == 1:
            w += 1
        if h % 2 == 1:
            h += 1
        return (w, h)
