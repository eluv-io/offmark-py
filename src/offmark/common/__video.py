import logging
import pprint

import ffmpeg

from .__logging import trace

logger = logging.getLogger(__name__)


@trace(logger)
def probe(video_file):
    info = ffmpeg.probe(video_file)
    logger.debug(pprint.pformat(info))

    video_info = next(s for s in info['streams'] if s['codec_type'] == 'video')
    d = {
        'width': int(video_info['width']),
        'height': int(video_info['height'])
    }
    logger.info(f'Probed video: {d}')

    return d
