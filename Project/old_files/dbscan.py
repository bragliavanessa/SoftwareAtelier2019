import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import time

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import image
from sklearn.metrics import pairwise_distances
np.set_printoptions(threshold=np.inf)

img = cv2.imread('./frames/frame0.png')
img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

def image_to_graph(img):
    graph = []
    for (x, row) in enumerate(img):
        for (y, col) in enumerate(row):
            [r, g, b] = img[x, y]
            graph.append([r, g, b])
    return np.array(graph)


graph = image_to_graph(img)

db = DBSCAN(eps=30, min_samples=90, metric = 'euclidean',algorithm ='auto')
db.fit(graph)
labels = db.labels_

K = len(set(labels)) - (1 if -1 in labels else 0)
labels = labels.reshape(img.shape[:2])


plt.figure(figsize=(5, 5))
# plt.imshow(img)
plt.imshow(labels*255/K, alpha=0.8)

for l in range(K):
    plt.contour(labels == l,
                colors=[plt.cm.nipy_spectral(l / float(K))])
plt.show()

