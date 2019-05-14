import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import kmeans as km
from skimage import filters
from skimage import measure


# original image
# -1 loads as-is so if it will be 3 or 4 channel as the original
# for i in range(100):
vidcap = cv2.VideoCapture('./video/ball.mp4')
success, image = vidcap.read(-1)
i = 0
# while i<1000:
#     success, image = vidcap.read(-1)
#     i+=1
while success:
    # name = './frames/frame'+str(i)+'.png'
    # image = cv2.imread(name, -1)
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[(319.913, 65.5129), (40.7903, 668.352),
                             (1279.5, 630.365), (909.539, 70.4677)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
    K = 5
    labels = km.kmeans(masked_image, K)

    for (x, row) in enumerate(masked_image):
        for (y, col) in enumerate(row):
            [r, g, b] = masked_image[x, y]
            if [r, g, b] == [0, 0, 0]:
                labels[x, y] = -1

    if not os.path.exists('./clusters/'):
        os.mkdir('./clusters/')
    j=0
    for k in range(K):
        idx = []
        for (x, row) in enumerate(masked_image):
            for (y, col) in enumerate(row):
                if labels[x, y] != k:
                    idx.append([x, y])
        idx = np.array(idx)
        new_img = np.copy(masked_image)
        new_img[idx[:, 0], idx[:, 1]] = [0, 0, 0]

        labels2 = km.kmeans(new_img, K)

        folder_name = './clusters/frame'+str(i)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
                
        for kk in range(K):
            file_name = './clusters/frame'+str(i)+'/cluster'+str(j)+'.png'
            idx2 = []
            for (x, row) in enumerate(new_img):
                for (y, col) in enumerate(row):
                    if labels2[x, y] != kk:
                        idx2.append([x, y])
            idx2 = np.array(idx2)
            new_img2 = np.copy(new_img)
            new_img2[idx2[:, 0], idx2[:, 1]] = [0, 0, 0]
            if os.path.exists(file_name):
                os.remove(file_name)
            cv2.imwrite(file_name, new_img2)
            j+=1
    success, image = vidcap.read()
    i += 1
    print(i)
