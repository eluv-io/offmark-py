import logging

import cv2
import numpy as np

from ..common.__logging import trace

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, frame_reader, frame_embedder, frame_writer):
        self.frame_reader = frame_reader
        self.frame_writer = frame_writer
        self.frame_embedder = frame_embedder

    @trace(logger)
    def start(self):
        while True:
            in_frame = self.frame_reader.read()
            if in_frame is None:
                logger.info('End of input stream')
                break

            out_frame = self.__mark_frame(in_frame)

            self.frame_writer.write(out_frame)

        self.frame_reader.close()
        self.frame_writer.close()
        logger.info('Done')

    def __mark_frame(self, frame_rgb):
        frame_yuv = cv2.cvtColor(frame_rgb.astype(np.float32), cv2.COLOR_BGR2YUV)
        wm_frame_yuv = self.frame_embedder.encode(frame_yuv)
        wm_frame_rgb = cv2.cvtColor(wm_frame_yuv, cv2.COLOR_YUV2BGR)
        wm_frame_rgb = np.clip(wm_frame_rgb, a_min=0, a_max=255)
        wm_frame_rgb = np.around(wm_frame_rgb).astype(np.uint8)
        return wm_frame_rgb
