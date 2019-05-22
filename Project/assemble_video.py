# Takes all the frames in the ./frame folder and assemble a video in the ./video folder

import cv2
import os
import re

# Need them to sort the alphanumeric string
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

# Declare folders
image_folder = './frames'
video_name = './video/result_video.mp4'

# Taking all the frames to assemble and sort them by name
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=natural_keys)


frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height),1)

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()