import scipy
import cv2
import sys
import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

fname='./clusters/frame50/cluster17.png'
blur_radius = 1.0
threshold = 50

img = scipy.misc.imread(fname)

img_color = cv2.imread(fname)

# smooth the image
imgf = ndimage.gaussian_filter(img, blur_radius)
threshold = 50

# find connected components
labeled, nr_objects = ndimage.label(imgf > threshold) 

count_balls = 0
idx=7
folder_name = './frames'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

for i in range(1,nr_objects+1):
    labels = labeled==i
    num = np.count_nonzero(labels)
    if num>750:
        count_balls+=1
        img_result = np.multiply(labels,img_color)

        if not os.path.exists('./frames/frame50'):
            os.mkdir(folder_name)

        filename = './frames/frame50/cluster'+str(idx)+'.png'
        if os.path.exists(filename):
            os.remove(filename)
        cv2.imwrite(filename, img_result)
        idx+=1
