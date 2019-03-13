# Take a video from the ./video folder and return the frames in the ./frames folder
# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

import cv2

vidcap = cv2.VideoCapture('./video/prova.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("./frames/frame%d.png" % count, image)    # Save the frame 
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1