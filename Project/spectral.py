import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import time

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.metrics import pairwise_distances
np.set_printoptions(threshold=np.inf)

img = cv2.imread('./frames/frame49.png')
img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img.shape[:2])
graph = []
for (x, row) in enumerate(img):
    for (y, col) in enumerate(row):
        [r, g, b] = img[x, y]
        graph.append([x, y, r, g, b])
graph = np.array(graph)
print('ciao')
n = len(graph)
sigma = 2*np.log(n)
# graph = image.img_to_graph(img)

# D = pairwise_distances(graph)
# row = []
# for x in graph:
#     row.append(np.linalg.norm(graph[0]-x))

# row = np.array(row)

# D = np.zeros()
dists = sp.spatial.distance.pdist(graph)
D = sp.spatial.distance.squareform(dists)


S = np.exp(-(D**2) / (2*(sigma**2)))
print(S.shape)

epsilon = 60
G = np.array([x < epsilon for x in D])
G = G.astype(int)

W = G * S
W = sp.sparse.csr_matrix(W)
# # beta = 30
# # eps = 1e-6
# # graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
print('a')
K = 5


labels = spectral_clustering(W, n_clusters=K,
                             assign_labels='kmeans', random_state=42, eigen_solver='amg')
labels = labels.reshape(img.shape[:2])

plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.imshow(labels*255/K, alpha=0.8)
# for l in range(K):
#     c = np.array([[np.array_equal(y, [l, l, l])
#                    for y in x]for x in labels])
#     plt.contour(c,
#                 colors=[plt.cm.nipy_spectral(l / float(K))])


for l in range(K):
    plt.contour(labels == l,
                colors=[plt.cm.nipy_spectral(l / float(K))])
plt.show()
