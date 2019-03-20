import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

img = cv2.imread('./frames/frame20.png')
img = cv2.resize(img, None, fx=0.02, fy=0.02, interpolation=cv2.INTER_CUBIC)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = img.reshape((-1, 3))

# convert to np.float32
# img = np.float32(img)

graph = image.img_to_graph(img)

beta = 10
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
# print(graph)

# Force the solver to be arpack, since amg is numerically
# unstable on this example
K = 10
# labels = spectral_clustering(graph, n_clusters=K, eigen_solver='amg')
# labels = labels.reshape(img.shape)
# labels = labels * (255/K)
# labels = labels.astype(int)


labels = spectral_clustering(graph, n_clusters=K,
                             assign_labels='kmeans', random_state=42, eigen_solver='amg')
labels = labels.reshape(img.shape)

plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.imshow(labels*255/K, alpha=0.8)
# for l in range(K):

#     plt.contour(z, colors=[plt.cm.nipy_spectral(l / float(K))])

plt.show()
