import scipy
import cv2
import sys
import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def conn_comp(img,imgidx):
    # fname='./clusters/frame50/cluster17.png'
    blur_radius = 1.0
    threshold = 50

    # img = scipy.misc.imread(fname)

    # img_color = cv2.imread(fname)

    # smooth the image
    imgf = ndimage.gaussian_filter(img, blur_radius)
    threshold = 50

    # find connected components
    labeled, nr_objects = ndimage.label(imgf > threshold) 

    count_balls = 0
    idx=7

    balls = []

    for i in range(1,nr_objects+1):
        labels = labeled==i
        num = np.count_nonzero(labels)
        if num>750:
            count_balls+=1
            img_result = np.multiply(labels,img)
            balls.append(img_result)
            idx+=1

    return [balls,count_balls]
    
