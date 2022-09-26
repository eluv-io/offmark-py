import numpy as np
import cv2

from src.encoder.dwt_dct_svd_encoder import *
from src.decoder.dwt_dct_svd_decoder import *

from src.encoder.dtcwt_key_encoder import *
from src.decoder.dtcwt_key_decoder import *

from src.encoder.dtcwt_img_encoder import *
from src.decoder.dtcwt_img_decoder import *

from src.generator.shuffler import *
from src.degenerator.de_shuffler import *

from src.generator.grayscale import *
from src.degenerator.de_grayscale import *

from src.generator.corr_shuffler import *
from src.degenerator.de_corr_shuffler import *

from src.generator.block_shuffler import *
from src.degenerator.de_block_shuffler import *

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
	DctEncoder()
]

gen_idx = 0
coder_idx = 2
generator = generators[gen_idx]
degenerator = degenerators[gen_idx]
encoder = encoders[coder_idx]
decoder = decoders[coder_idx]

payload = None

# payload is read in differently depending on the input type (char or image)
wm_type = generator.wm_type()
if wm_type == "bits":
	payload = np.array([0,1,1,0,0,1,0,1])
	print("Payload: ", payload)
elif wm_type == "grayscale":
	wm_path = "wms/qr.jpeg"
	wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
	assert wm is not None, "Watermark not found in {}".format(wm_path)
	payload = wm
	print("Payload: ", wm_path)

path = "imgs/frame63.jpeg"
output_path = "output.jpeg"

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

bgr = cv2.imread(output_path)
yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

# decode
decoded_wm = decoder.decode(yuv)

# degenerate
ret_payload = degenerator.set_shape(payload.shape).degenerate(decoded_wm)
print("Decoded :", ret_payload)
cv2.imwrite("degenerate.jpeg", ret_payload)

# a = (payload - np.mean(payload)) / (np.std(payload) * len(payload))
# b = (ret_payload - np.mean(ret_payload)) / (np.std(ret_payload))
# c = np.correlate(a, b, 'full')
# print("Maximum correlation: ", np.amax(c))
