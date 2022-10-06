import logging

import cv2
import numpy as np

from ..common.__logging import trace

logger = logging.getLogger(__name__)


class Extractor:
    def __init__(self, frame_reader, frame_extractor, degenerator):
        self.frame_reader = frame_reader
        self.frame_extractor = frame_extractor
        self.degenerator = degenerator

    @trace(logger)
    def start(self):
        while True:
            in_frame = self.frame_reader.read()
            if in_frame is None:
                logger.info('End of input stream')
                break

            self.__check_frame(in_frame)

        self.frame_reader.close()
        logger.info('Done')

    def __check_frame(self, frame_rgb):
        wm_frame_yuv = cv2.cvtColor(frame_rgb.astype(np.float32), cv2.COLOR_BGR2YUV)
        frame_yuv = self.frame_extractor.decode(wm_frame_yuv)
        out = self.degenerator.degenerate(frame_yuv)
        logger.info(out)
