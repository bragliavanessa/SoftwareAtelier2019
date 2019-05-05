import cv2
import numpy as np
import matplotlib.pyplot as plt

# original image
# -1 loads as-is so if it will be 3 or 4 channel as the original
image = cv2.imread('./frames/frame0.png', -1)
# mask defaulting to black for 3-channel and transparent for 4-channel
# (of course replace corners with yours)
mask = np.zeros(image.shape, dtype=np.uint8)
roi_corners = np.array([[(319.913,65.5129), (40.7903,668.352), (1279.5,630.365), (909.539,70.4677)]], dtype=np.int32)
# fill the ROI so it doesn't get wiped out when the mask is applied
channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(mask, roi_corners, ignore_mask_color)
# from Masterfool: use cv2.fillConvexPoly if you know it's convex

# apply the mask
masked_image = cv2.bitwise_and(image, mask)

plt.figure(figsize=(5, 5))
plt.imshow(masked_image)
plt.show()

# save the result
cv2.imwrite('./frames/a_masked.png', masked_image)