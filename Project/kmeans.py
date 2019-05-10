import numpy as np
import cv2
import matplotlib.pyplot as plt

# img = cv2.imread('./frames/a_masked.png')
# img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)


def image_to_graph(img):
    graph = []
    for (x, row) in enumerate(img):
        for (y, col) in enumerate(row):
            [r, g, b] = img[x, y]
            graph.append([r, g, b])
    return np.array(graph)


def kmeans(img, K):
    Z = image_to_graph(img)
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = label.reshape(img.shape[:2])
    return labels
