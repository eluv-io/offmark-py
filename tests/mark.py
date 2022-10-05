import logging
import os

import numpy as np

from offmark.embed.dwt_dct_svd_encoder import DwtDctSvdEncoder
from offmark.generator.shuffler import Shuffler
from offmark.video.embedder import Embedder
from offmark.video.frame_reader import FileDecoder
from offmark.video.frame_writer import FileEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s')


def run():
    this_dir = os.path.dirname(__file__)
    in_file = os.path.join(this_dir, 'media', 'in.mp4')
    out_file = os.path.join(this_dir, 'out', 'marked.mp4')
    payload = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    print("Payload: ", payload)

    r = FileDecoder(in_file)
    w = FileEncoder(out_file, r.width, r.height)

    # Initialize Frame Embedder
    frame_embedder = DwtDctSvdEncoder()
    capacity = frame_embedder.wm_capacity((r.height, r.width, 3))

    # Initialize Generator
    generator = Shuffler(key=0)
    wm = generator.generate_wm(payload, capacity)
    frame_embedder.read_wm(wm)

    # Start watermarking and transcoding
    # TODO properly preserve the original video encoding and container
    video_embedder = Embedder(r, frame_embedder, w)
    video_embedder.start()


# TODO CLI flags to choose embedder
if __name__ == '__main__':
    run()
