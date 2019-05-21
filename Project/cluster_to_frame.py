import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def balls_to_frames(balls, numred, numyellow):
    img=0
    for ball in balls:
        b, g, r = cv2.split(ball)
        rgb = [b,g,r]
        dst = cv2.merge(rgb,3)

        img+=dst

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (1000,35)
    fontScale              = 1
    fontColor              = (50,50,200)
    lineType               = 2

    number_of_Red = str(numred)+' red balls'
    cv2.putText(img,number_of_Red, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    number_of_Yellow = str(numyellow)+' yellow balls'
    bottomLeftCornerOfText = (1000,80)
    fontColor              = (0,200,255)

    cv2.putText(img,number_of_Yellow, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.imwrite("./frames_WC/ciaone.png", img)
    return img


