import numpy as np
import cv2
import dtcwt

class DtcwtKeyEncoder:

    def __init__(self, key=None, str=1.0, step=5.0):
        self.key = key
        default_scale = 10.0
        self.alpha = default_scale * str
        self.step = step

    def read_wm(self, wm):
        wm_transform = dtcwt.Transform2d()
        wm_coeffs = wm_transform.forward(wm, nlevels=1)
        self.wm_coeffs = wm_coeffs

    def wm_capacity(self, frame_shape):
        (h, w) = self.__infer_wm_shape(frame_shape)
        return (h, w)

    def encode(self, yuv):
        yuv_transform = dtcwt.Transform2d()
        yuv_coeffs = yuv_transform.forward(yuv[:, :, 1], nlevels=3)
        y_transform = dtcwt.Transform2d()
        y_coeffs = y_transform.forward(yuv[:, :, 0], nlevels=3)

        # Masks for level 3 subbands
        masks3 = [0 for i in range(6)]
        shape3 = y_coeffs.highpasses[2][:, :, 0].shape
        for i in range(6):
            masks3[i] = cv2.filter2D(np.abs(y_coeffs.highpasses[1][:,:,i]), -1, np.array([[1/4, 1/4], [1/4, 1/4]]))
            masks3[i] = np.ceil(self.rebin(masks3[i], shape3) / self.step)
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

    def __infer_wm_shape(self, img_shape):
        h = (((img_shape[0] + 1) // 2 + 1) // 2 + 1) // 2
        w = (((img_shape[1] + 1) // 2 + 1) // 2 + 1) // 2
        if h % 2 == 1:
            h += 1
        if w % 2 == 1:
            w += 1
        return (h, w)

    def rebin(self, a, shape):
        if a.shape[0] % 2 == 1:
            a = np.vstack((a, np.zeros((1, a.shape[1]))))
        sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
        return a.reshape(sh).mean(-1).mean(1)
