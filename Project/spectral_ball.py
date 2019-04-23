import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import time

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.metrics import pairwise_distances
np.set_printoptions(threshold=np.inf)

img = cv2.imread('./frames/frame00.png')
img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)

def image_to_graph_xy(img):
    graph_xy = []
    for (x, row) in enumerate(img):
        for (y, col) in enumerate(row):
            # [r, g, b] = img[x, y]
            graph_xy.append([x, y])
    return np.array(graph_xy)

def image_to_graph_rgb(img,labels):
    graph_rgb = []
    for (x, row) in enumerate(img):
        for (y, col) in enumerate(row):
            [r, g, b] = img[x, y]
            l = labels[x,y]
            graph_rgb.append([r, g, b, l])
    return np.array(graph_rgb)


def kNNSimGraph(D):
    # k = int(np.ceil(2*np.log(len(D))));
    k = 150
    M = np.zeros(D.shape)
    for (index, row) in enumerate(D):
        sort_idx = np.argsort(row)[:k]
        M[index][sort_idx] = 1
    return M


graph_xy = image_to_graph_xy(img)

n = len(graph_xy)
sigma_xy = 2*np.log(n)

dists_xy = sp.spatial.distance.pdist(graph_xy)

D_xy = sp.spatial.distance.squareform(dists_xy)

S_xy = np.exp(-(D_xy**2) / (2*(sigma_xy**2)))


epsilon = 20
G = np.array([x < epsilon for x in D_xy])
G = G.astype(int)


W = G * S_xy
W = sp.sparse.csr_matrix(W)

K = 3


labels = spectral_clustering(
    W, n_clusters=K, assign_labels='kmeans', random_state=42, eigen_solver='amg')
labels = labels.reshape(img.shape[:2])


sigma_rgb = 2*np.log(n)*4
graph_rgb = image_to_graph_rgb(img,labels)


dists_rgb = sp.spatial.distance.pdist(graph_rgb)

D_rgb = sp.spatial.distance.squareform(dists_rgb)

S_rgb = np.exp(-(D_rgb**2) / (2*(sigma_rgb**2)))


epsilon = 80
G_rgb = np.array([x < epsilon for x in D_rgb])
G_rgb = G_rgb.astype(int)


Wrgb = G_rgb * S_rgb
Wrgb = sp.sparse.csr_matrix(Wrgb)

labels = spectral_clustering(
    Wrgb, n_clusters=K, assign_labels='kmeans', random_state=42, eigen_solver='amg')
labels = labels.reshape(img.shape[:2])




plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.imshow(labels*255/K, alpha=0.8)

for l in range(K):
    plt.contour(labels == l,
                colors=[plt.cm.nipy_spectral(l / float(K))])
plt.show()
