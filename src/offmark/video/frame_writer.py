import logging
import subprocess

import ffmpeg
import numpy as np

from ..common.__logging import trace

logger = logging.getLogger(__name__)


class FrameWriter:
    def __init__(self):
        pass

    def write(self, frame: np.ndarray):
        pass

    def close(self):
        pass


class FileEncoder(FrameWriter):
    def __init__(self, file, width, height):
        super().__init__()
        self.file = file
        self.__start_ffmpeg(width, height)

    @trace(logger)
    def __start_ffmpeg(self, width, height):
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .output(self.file, pix_fmt='yuv420p')
            .overwrite_output()
            .global_args('-loglevel', 'quiet')
            .compile()
        )
        self.ffmpeg = subprocess.Popen(args, stdin=subprocess.PIPE)

    def write(self, frame):
        self.ffmpeg.stdin.write(
            frame.astype(np.uint8).tobytes()
        )

    @trace(logger)
    def close(self):
        logger.info('Waiting for ffmpeg encoder')
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
