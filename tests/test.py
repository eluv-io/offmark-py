import os

import cv2
import numpy as np

from offmark.degenerator.de_block_shuffler import DeBlockShuffler
from offmark.degenerator.de_corr_shuffler import DeCorrShuffler
from offmark.degenerator.de_grayscale import DeGrayScale
from offmark.degenerator.de_shuffler import DeShuffler
from offmark.embed.dct_encoder import DctEncoder
from offmark.embed.dtcwt_img_encoder import DtcwtImgEncoder
from offmark.embed.dtcwt_key_encoder import DtcwtKeyEncoder
from offmark.embed.dwt_dct_svd_encoder import DwtDctSvdEncoder
from offmark.extract.dct_decoder import DctDecoder
from offmark.extract.dtcwt_img_decoder import DtcwtImgDecoder
from offmark.extract.dtcwt_key_decoder import DtcwtKeyDecoder
from offmark.extract.dwt_dct_svd_decoder import DwtDctSvdDecoder
from offmark.generator.block_shuffler import BlockShuffler
from offmark.generator.corr_shuffler import CorrShuffler
from offmark.generator.grayscale import GrayScale
from offmark.generator.shuffler import Shuffler

this_dir = os.path.dirname(__file__)

key1 = 0

generators = [
    Shuffler(key=key1),
    GrayScale(key=key1),
    CorrShuffler(key=key1),
    BlockShuffler(key=key1)
]

degenerators = [
    DeShuffler(key=key1),
    DeGrayScale(key=key1),
    DeCorrShuffler(key=key1),
    DeBlockShuffler(key=key1)
]

encoders = [
    DwtDctSvdEncoder(),
    DtcwtKeyEncoder(),
    DtcwtImgEncoder(),
    DctEncoder()
]

decoders = [
    DwtDctSvdDecoder(),
    DtcwtKeyDecoder(),
    DtcwtImgDecoder(),
    DctDecoder()
]

# gen_idx:coder_idx combinations: 0:0, 0:3, 1:0, 1:3, 2:1, 3:2
gen_idx = 2
coder_idx = 1
generator = generators[gen_idx]
degenerator = degenerators[gen_idx]
encoder = encoders[coder_idx]
decoder = decoders[coder_idx]

payload = None

# payload is read in differently depending on the input type (char or image)
wm_type = generator.wm_type()
if wm_type == "bits":
    payload = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    print("Payload: ", payload)
elif wm_type == "grayscale":
    wm_path = os.path.join(this_dir, 'media', 'wms', 'qr.jpeg')
    wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    assert wm is not None, "Watermark not found in {}".format(wm_path)
    payload = wm
    print("Payload: ", wm_path)

path = os.path.join(this_dir, 'media', 'imgs', 'frame63.jpeg')
output_path = os.path.join(this_dir, 'out', 'output.jpeg')

# Read a frame and convert to YUV pixel format
img = cv2.imread(path)
assert img is not None, "Image not found in {}".format(path)
bgr = img.astype(np.float32)
yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

# generate
wm = generator.generate_wm(payload, encoder.wm_capacity(yuv.shape))

# encode
encoder.read_wm(wm)
yuv = encoder.encode(yuv)
wmed_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
wmed_frame = np.clip(wmed_frame, a_min=0, a_max=255)
wmed_frame = np.around(wmed_frame).astype(np.uint8)
cv2.imwrite(output_path, wmed_frame)

# diff with original (scaled for visibility)
diff = img.astype(np.int32) - wmed_frame.astype(np.int32)
diff = np.abs(diff)
diff_max = np.max(diff)
diff = np.multiply(diff, 255 * 3/diff_max)
diff = diff.clip(0, 255)
diff_path = os.path.join(this_dir, 'out', 'diff.jpeg')
cv2.imwrite(diff_path, diff)

bgr = cv2.imread(output_path).astype(np.float32)
yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

# decode
decoded_wm = decoder.decode(yuv)

# degenerate
ret_payload = degenerator.set_shape(payload.shape).degenerate(decoded_wm)
print("Decoded:", ret_payload)
cv2.imwrite(os.path.join(this_dir, 'out', 'degenerate.jpeg'), ret_payload)

# a = (payload - np.mean(payload)) / (np.std(payload) * len(payload))
# b = (ret_payload - np.mean(ret_payload)) / (np.std(ret_payload))
# c = np.correlate(a, b, 'full')
# print("Maximum correlation: ", np.amax(c))
