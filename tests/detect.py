import logging
import os

import numpy as np

from offmark.degenerator.de_shuffler import DeShuffler
from offmark.extract.dwt_dct_svd_decoder import DwtDctSvdDecoder
from offmark.video.extractor import Extractor
from offmark.video.frame_reader import FileDecoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s')


def run():
    this_dir = os.path.dirname(__file__)
    in_file = os.path.join(this_dir, 'out', 'marked.mp4')
    payload = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    print("Payload: ", payload)

    r = FileDecoder(in_file)

    degenerator = DeShuffler(key=0)
    degenerator.set_shape(payload.shape)

    frame_extractor = DwtDctSvdDecoder()

    video_extractor = Extractor(r, frame_extractor, degenerator)
    video_extractor.start()


if __name__ == '__main__':
    run()
