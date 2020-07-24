#convert -gravity center -background white -extent 2560x1600 -delay 1 -loop 0 gif/*.dot.png gif/animated.gif
#convert -background white -gravity center -delay 1 -loop 0 gif/*.dot.png gif/animated.gif
OUTPUT=./gif/animated.mp4

rm -f $OUTPUT

ffmpeg \
  -y \
  -framerate 0.5 \
  -pattern_type glob \
  -i 'gif/*.dot.png' \
  -r 5 \
  -vcodec mpeg4 \
  $OUTPUT 
 # -framerate 0.5 \
 # -vf scale=1024:-1 \
 # $OUTPUT \

