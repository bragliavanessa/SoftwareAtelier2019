import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


c=0
img=0
DIR='./frames/frame3'
num_files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
print(num_files)


while c<num_files:
    file_name = "./frames/frame3/cluster"+str(c)+".png"



    src = cv2.imread(file_name, 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(src)
    rgb = [b,g,r]
    dst = cv2.merge(rgb,3)

    img+=dst

    c+=1

cv2.imwrite("./frames_WC/test.png", img)


