import logging
import subprocess

import ffmpeg
import numpy as np

from ..common.__logging import trace
from ..common.__video import probe

logger = logging.getLogger(__name__)


# TODO extend ABC (abstract base class)?
class FrameReader:
    def __init__(self):
        pass

    def read(self) -> np.ndarray:
        """Read one frame in RGB format."""
        pass

    def close(self):
        pass


# TODO FileDecoder in separate file?
# TODO pix_fmt yuv420p?
class FileDecoder(FrameReader):
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.__start_ffmpeg()

    @trace(logger)
    def __start_ffmpeg(self):
        # Width and height in pixels
        info = probe(self.file)
        self.width = info['width']
        self.height = info['height']

        # RGB24
        self.frame_size_bytes = self.width * self.height * 3

        args = (
            ffmpeg
            .input(self.file)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args('-loglevel', 'quiet')
            .compile()
        )
        self.ffmpeg = subprocess.Popen(args, stdout=subprocess.PIPE)

    def read(self):
        frame_bytes = self.ffmpeg.stdout.read(self.frame_size_bytes)
        if len(frame_bytes) == 0:
            frame = None
        else:
            assert len(frame_bytes) == self.frame_size_bytes
            frame = (
                np
                .frombuffer(frame_bytes, np.uint8)
                .reshape(self.height, self.width, 3)
            )
        return frame

    @trace(logger)
    def close(self):
        logger.info('Waiting for ffmpeg decoder')
        self.ffmpeg.wait()
