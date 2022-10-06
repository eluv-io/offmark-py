import cv2
import numpy as np

class DctDecoder:

    def __init__(self, key=None, alpha=20):
        self.key = key
        self.alpha = alpha

    def decode(self, yuv):
        blk_shape = (8, 8)
        channel = yuv[:,:,1]
        lum_mask = self.luminance_mask(yuv[:,:,0])
        tex_mask = self.texture_mask(yuv[:,:,0])
        mask = tex_mask * lum_mask
        c = 0
        wm = np.zeros(yuv.shape[0] * yuv.shape[1] // blk_shape[0] // blk_shape[1])
        for i in range(channel.shape[0] // blk_shape[0]):
            for j in range(channel.shape[1] // blk_shape[1]):
                blk = channel[i * blk_shape[0] : i * blk_shape[0] + blk_shape[0],
                                j * blk_shape[1] : j * blk_shape[1] + blk_shape[1]]
                step = self.alpha * mask[i][j]
                coeffs = cv2.dct(blk)
                wm_bit = int(np.around(coeffs[2][1] / step) % 2 == 1)
                wm[c] = wm_bit
                c += 1
        return np.array(wm).reshape(1, -1)

    def luminance_mask(self, lum):
        blk_shape = (8, 8)
        rows = lum.shape[0] // blk_shape[0]
        cols = lum.shape[1] // blk_shape[1]
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                blk = lum[i * blk_shape[0]:i * blk_shape[0] + blk_shape[0],
                          j * blk_shape[1]:j * blk_shape[1] + blk_shape[1]]
                coeffs = cv2.dct(blk)
                mask[i][j] = coeffs[0][0]
        l_min, l_max = 90, 255
        f_max = 2
        mask /= 8
        mean = max(l_min, np.mean(mask))
        f_ref = 1 + (mean - l_min) * (f_max - 1) / (l_max - l_min)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] > mean:
                    mask[i][j] = 1 + (mask[i][j] - mean) / (l_max - mean) * (f_max - f_ref)
                elif mask[i][j] < 15:
                    mask[i][j] = 1.25
                elif mask[i][j] < 25:
                    mask[i][j] = 1.125
                else:
                    mask[i][j] = 1
        return mask

    def texture_mask(self, lum):
        blk_shape = (8, 8)
        rows = lum.shape[0] // blk_shape[0]
        cols = lum.shape[1] // blk_shape[1]
        mask = np.full((rows, cols), 1.0)
        for i in range(rows):
            for j in range(cols):
                blk = lum[i * blk_shape[0]:i * blk_shape[0] + blk_shape[0],
                          j * blk_shape[1]:j * blk_shape[1] + blk_shape[1]]
                coeffs = cv2.dct(blk)
                coeffs = np.abs(coeffs)
                dcl = coeffs[0][0] + coeffs[0][1] + coeffs[0][2] + coeffs[1][0] + coeffs[1][1] + coeffs[2][0]
                eh = np.sum(coeffs) - dcl
                if eh > 125:
                    e = coeffs[3][0] + coeffs[4][0] + coeffs[5][0] + coeffs[6][0] + \
                        coeffs[0][3] + coeffs[0][4] + coeffs[0][5] + coeffs[0][6] + \
                        coeffs[2][1] + coeffs[1][2] + coeffs[2][2] + coeffs[3][3]
                    h = eh - e
                    l = dcl - coeffs[0][0]
                    a1, b1 = 2.3, 1.6
                    a2, b2 = 1.4, 1.1
                    l_e, le_h  = l / e, (l + e) / h
                    if eh > 900:
                        if (l_e  >= a2 and le_h >= b2) or (l_e >= b2 and le_h >= a2) or le_h > 4:
                            mask[i][j] = 1.125 if l + e <= 400 else 1.25
                        else:
                            mask[i][j] = 1 + 1.25 * (eh - 290) / (1800 - 290)
                    else:
                        if (l_e  >= a1 and le_h >= b1) or (l_e >= b1 and le_h >= a1) or le_h > 4:
                            mask[i][j] = 1.125 if l + e <= 400 else 1.25
                        elif e + h > 290:
                            mask[i][j] = 1 + 1.25 * (eh - 290) / (1800 - 290)
        return mask
