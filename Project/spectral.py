import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import time

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.metrics import pairwise_distances
np.set_printoptions(threshold=np.inf)

img = cv2.imread('./frames/frame102.png')
img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def image_to_graph(img):
    graph_xy = []
    graph_rgb = []
    for (x, row) in enumerate(img):
        for (y, col) in enumerate(row):
            [r, g, b] = img[x, y]
            graph_xy.append([x, y])
            graph_rgb.append([r, g, b])
            # graph.append([x/img.shape[1], y/img.shape[0], r/255, g/255, b/255])
    return [np.array(graph_xy), np.array(graph_rgb)]


def kNNSimGraph(D):
    # k = int(np.ceil(2*np.log(len(D))));
    k = 150
    M = np.zeros(D.shape)
    for (index, row) in enumerate(D):
        sort_idx = np.argsort(row)[:k]
        M[index][sort_idx] = 1
    return M


[graph_xy, graph_rgb] = image_to_graph(img)
n = len(graph_xy)
sigma_xy = 2*np.log(n)
sigma_rgb = 2*np.log(n)*4

dists_xy = sp.spatial.distance.pdist(graph_xy)
dists_rgb = sp.spatial.distance.pdist(graph_rgb)
D_xy = sp.spatial.distance.squareform(dists_xy)
D_rgb = sp.spatial.distance.squareform(dists_rgb)

D = D_xy+D_rgb


S_xy = np.exp(-(D_xy**2) / (2*(sigma_xy**2)))
S_rgb = np.exp(-(D_rgb**2) / (2*(sigma_rgb**2)))


S = S_xy + S_rgb

# epsilon = np.max(sp.sparse.csgraph.minimum_spanning_tree(S).toarray())
epsilon = 100
G = np.array([x < epsilon for x in D])
G = G.astype(int)
# G = kNNSimGraph(D)

W = G * S
W = sp.sparse.csr_matrix(W)

K = 3


labels = spectral_clustering(
    W, n_clusters=K, assign_labels='kmeans', random_state=42, eigen_solver='amg')
labels = labels.reshape(img.shape[:2])

plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.imshow(labels*255/K, alpha=0.8)

for l in range(K):
    plt.contour(labels == l,
                colors=[plt.cm.nipy_spectral(l / float(K))])
plt.show()
