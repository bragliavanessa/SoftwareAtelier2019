import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import time

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.metrics import pairwise_distances
np.set_printoptions(threshold=np.inf)

img = cv2.imread('./frames/frame2.png')
img = cv2.resize(img, None, fx=0.08, fy=0.08, interpolation=cv2.INTER_CUBIC)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def remove_background(img):
    #== Parameters           
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 100
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0,0.0,0.0) # In BGR format


    # Read image
    # img = cv2.imread('./frames/frame0.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Edge detection 
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # Find contours
    contour_info = []
    contours,hierachy=cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]


    # Create empty mask, draw filled polygon on it corresponding to largest contour
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)

    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))

    # Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    # Blend masked img into MASK_COLOR background
    mask_stack  = mask_stack.astype('float32') / 255.0   
    # print(mask_stack)
    # cv2.imshow('img', mask_stack)
    # cv2.waitKey()      
    img = img.astype('float32') / 255.0   
    # print(mask_stack[1])
    # print(img[1])
    graph_xy = []
    graph_idx = []
    for (x, row) in enumerate(img):
        for (y, col) in enumerate(row):
            # print(mask_stack[x,y])
            # print(np.any(mask_stack[x,y]))
            if np.any(mask_stack[x,y]):
                [r, g, b] = img[x, y]
                graph_xy.append([x, y, r, g, b])
                graph_idx.append([x, y])
                # graph_rgb.append([r, g, b])
                # graph.append([x/img.shape[1], y/img.shape[0], r/255, g/255, b/255])

    return [np.array(graph_xy),np.array(graph_idx)]

def image_to_graph(img):
    graph_xy = []
    graph_rgb = []
    for (x, row) in enumerate(img):
        for (y, col) in enumerate(row):
            [r, g, b] = img[x, y]
            graph_xy.append([x, y, r, g, b])
            # graph_rgb.append([r, g, b])
            # graph.append([x/img.shape[1], y/img.shape[0], r/255, g/255, b/255])
    return np.array(graph_xy)


def kNNSimGraph(D):
    # k = int(np.ceil(2*np.log(len(D))));
    k = 150
    M = np.zeros(D.shape)
    for (index, row) in enumerate(D):
        sort_idx = np.argsort(row)[:k]
        M[index][sort_idx] = 1
    return M


# graph_xy = image_to_graph(img)
[graph_xy, graph_idx] = remove_background(img)
w = np.diag([0.1,0.1,0.3,0.3,0.3])
graph_xy = np.matmul(graph_xy,w)


n = len(graph_xy)
sigma_xy = 2*np.log(n)
# sigma_rgb = 2*np.log(n)*4

dists_xy = sp.spatial.distance.pdist(graph_xy)

# dists_rgb = sp.spatial.distance.pdist(graph_rgb)
D_xy = sp.spatial.distance.squareform(dists_xy)
# D_rgb = sp.spatial.distance.squareform(dists_rgb)

D = D_xy #+D_rgb


S_xy = np.exp(-(D_xy**2) / (2*(sigma_xy**2)))

# print('ciao')
# S_rgb = np.exp(-(D_rgb**2) / (2*(sigma_rgb**2)))


S = S_xy #+ S_rgb

# print(S[0])
# exit()

# epsilon = np.max(sp.sparse.csgraph.minimum_spanning_tree(S).toarray())
# print(epsilon)
epsilon = 100
G = np.array([x < epsilon for x in D])
# print(G[0])
G = G.astype(int)
# G = kNNSimGraph(D)

W = G * S
W = sp.sparse.csr_matrix(W)

K = 3


labels = spectral_clustering(
    W, n_clusters=K, assign_labels='kmeans', random_state=42, eigen_solver='arpack')

pos_l = np.array([-1]*n)
pos_l = pos_l.reshape(img.shape[:2])
pos_l[graph_idx[:,0],graph_idx[:,1]] = labels

labels = pos_l


plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.imshow(labels*255/K, alpha=0.8)

for l in range(K):
    plt.contour(labels == l,
                colors=[plt.cm.nipy_spectral(l / float(K))])
plt.show()
