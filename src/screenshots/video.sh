#!/bin/bash

# USE
# -1) sudo apt install mpv
# 0) sudo apt install ffmpeg
# 1) chmod +x ./run_mp.sh
# 2_ run the script to create "video.mp4" in the same directory: $ ./run_mp.sh

FRAME_RATE=5
RESOLUTION=640x480

ffmpeg -r $FRAME_RATE -f image2 -s $RESOLUTION -i frame_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p video.mp4
rm *.png
mpv video.mp4 