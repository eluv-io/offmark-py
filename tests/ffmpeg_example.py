import ffmpeg

(
    ffmpeg
    .input('tests/media/in.mp4')
    .hflip()
    .output('tests/out/ffmpeg_example.mp4')
    .run()
)
