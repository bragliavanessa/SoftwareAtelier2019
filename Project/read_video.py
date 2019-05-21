import cv2
import os


vidcap = cv2.VideoCapture('./video/ball.mp4')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("./frames/frame%d.png" % count, image)    # Save the frame
    success, image = vidcap.read()
    count += 1