import cv2
import numpy as np
from conn_components import conn_comp
from cluster_to_frame import balls_to_frames
import os
import sys
import matplotlib.pyplot as plt
import kmeans as km
from skimage import filters
from skimage import measure
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from CNN.Model import Net
from CNN.Dataset import ImageDataset
from skimage import io
from skimage.transform import resize


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.load_state_dict(torch.load("./CNN/cnn", 'cpu'))
net.eval()
net.to(device)

DIR = "./frames/"
num_files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


i=sys.argv[1]

file_name = "./frames/frame"+str(i)+".png"
image = cv2.imread(file_name)
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
y=0
labels = km.kmeans(masked_image, K)
yellow2=0
redballs=0
yellowballs=0

for (x, row) in enumerate(masked_image):
    for (y, col) in enumerate(row):
        [r, g, b] = masked_image[x, y]
        if [r, g, b] == [0, 0, 0]:
            labels[x, y] = -1

if not os.path.exists('./clusters/'):
    os.mkdir('./clusters/')
j = 0
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

    for kk in range(K):
        # file_name = './clusters/frame'+str(i)+'_cluster'+str(j)+'.png'
        idx2 = []
        for (x, row) in enumerate(new_img):
            for (y, col) in enumerate(row):
                if labels2[x, y] != kk:
                    idx2.append([x, y])
        idx2 = np.array(idx2)
        new_img2 = np.copy(new_img)
        new_img2[idx2[:, 0], idx2[:, 1]] = [0, 0, 0]
        image_t = resize(new_img2, (256, 256))
        image_t = image_t.transpose((2, 0, 1))
        image_t = torch.tensor([image_t])
        image_t = image_t.float()
        outputs = net(image_t)
        _, predicted = torch.max(outputs, 1)
        pr = predicted.item()
        if pr==1:
            [redballs,num_red] = conn_comp(new_img2,i)
        elif pr==0:
            if y==1:
                print('2')
                [yellow2,num_yellow2] = conn_comp(new_img2,i)
            else:
                y = 1
                [yellowballs,num_yellow] = conn_comp(new_img2,i)

        j += 1


if int(i)<51:
    num_red=7
    num_yellow=7

if yellowballs and redballs:
    totalballs = np.concatenate((redballs, yellowballs))
elif yellowballs:
    totalballs = yellowballs
else:
    totalballs = redballs

if yellow2:
    totalballs = np.concatenate((totalballs, yellow2))

res = balls_to_frames(totalballs, num_red, num_yellow)