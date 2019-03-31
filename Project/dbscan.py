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
            # graph.append([x, y])
            graph.append([r, g, b])
            # graph.append([x, y, r, g, b])
    return np.array(graph)


graph = image_to_graph(img)
# n = len(graph_xy)
# sigma_xy = 2*np.log(n)
# sigma_rgb = 2*np.log(n)*4

# dists_xy = sp.spatial.distance.pdist(graph_xy)
# dists_rgb = sp.spatial.distance.pdist(graph_rgb)
# D_xy = sp.spatial.distance.squareform(dists_xy)
# D_rgb = sp.spatial.distance.squareform(dists_rgb)

# D = D_xy+D_rgb


# S_xy = np.exp(-(D_xy**2) / (2*(sigma_xy**2)))
# S_rgb = np.exp(-(D_rgb**2) / (2*(sigma_rgb**2)))


# S = S_rgb

# # epsilon = np.max(sp.sparse.csgraph.minimum_spanning_tree(S).toarray())
# epsilon = 100
# G = np.array([x < epsilon for x in D])
# G = G.astype(int)
# # G = kNNSimGraph(D) 

# W = G * S
# W = sp.sparse.csr_matrix(W)


# #############################################################################
# Generate sample data

# #############################################################################
# Compute DBSCAN
# db = DBSCAN(eps=0.3, min_samples=10).fit(W)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

db = DBSCAN(eps=30, min_samples=90, metric = 'euclidean',algorithm ='kd_tree')
db.fit(graph)
labels = db.labels_

K = len(set(labels)) - (1 if -1 in labels else 0)
print(K)
labels = labels.reshape(img.shape[:2])



# Number of clusters in labels, ignoring noise if present.


plt.figure(figsize=(5, 5))
# plt.imshow(img)
plt.imshow(labels*255/K, alpha=0.8)

for l in range(K):
    plt.contour(labels == l,
                colors=[plt.cm.nipy_spectral(l / float(K))])
plt.show()

