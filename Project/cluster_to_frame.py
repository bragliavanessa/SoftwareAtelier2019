import cv2
import numpy as np
import matplotlib.pyplot as plt

c=1
img=0

while c<5:
    file_name = "./clusters/frame0/cluster"+str(c)+".png"

    src = cv2.imread(file_name, 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(src)
    rgb = [b,g,r]
    dst = cv2.merge(rgb,3)

    img+=dst

    c+=1


# file_name = "./clusters/frame0/cluster4.png"

# src = cv2.imread(file_name, 1)
# tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# # _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
# b, g, r = cv2.split(src)
# rgb = [b,g,r]
# dst = cv2.merge(rgb,3)
# redImg = np.zeros(dst.shape, dst.dtype)
# redImg[:,:] = (0, 0, 0)
# redMask = cv2.bitwise_and(redImg, dst)
# cv2.addWeighted(redMask, 1, dst, 1, 0, dst)


# plt.imshow(dst)
# plt.show()

# img+=dst

cv2.imwrite("./frames_WC/test.png", img)


