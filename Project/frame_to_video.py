import cv2
import os

image_folder = './Frames'
video_name = './Video/siblings.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height),1)

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()