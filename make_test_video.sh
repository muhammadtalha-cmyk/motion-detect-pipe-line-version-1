#!/bin/sh
# Makes test.mp4 and test2.mp4 (20 s, still scene, red square moves at different times).
mk() { # name  motion-expression  y
ffmpeg -y -hide_banner -loglevel error -filter_complex \
"color=c=gray:s=1280x720:r=15:d=20[bg];color=c=red:s=240x240:r=15:d=20[fg];[bg][fg]overlay=x='$2':y=$3,format=yuv420p" \
-t 20 -c:v libx264 "$1" && echo "created $1"
}
mk test.mp4  'if(between(t,4,8),(t-4)*300,if(between(t,13,16),(16-t)*300,-200))' 240
mk test2.mp4 'if(between(t,9,13),(13-t)*300,if(between(t,1,3),(t-1)*400,-200))' 100
