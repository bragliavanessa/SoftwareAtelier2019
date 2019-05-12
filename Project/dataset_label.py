import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
# matplotlib.use("TkAgg")


labels = {
    "left": "Yellow",
    "right": "Red",
    "up": "White",
    " ": "Table",
    "z": "Noise",
}


def press(key, frame, cluster):
    sys.stdout.flush()
    print(key)
    if key == "x":
        sys.exit()
    if key in labels.keys():
        if os.path.exists("dataset.csv"):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        with open('dataset.csv', append_write) as fd:
            fd.write("frame%d/cluster%d\t%s\n" % (i, j, labels[key]))
            plt.close()
    else:
        print("Not a valid cluster")


start = 0
end = 3656
for i in range(start, end+1):
    for j in range(5):
        img = plt.imread("clusters/frame%d/cluster%d.png" % (i, j))

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect(
            'key_press_event', lambda event: press(event.key, i, j))
        # fig.canvas.manager.window.wm_geometry("+%d+%d" % (600, 250))
        fig.suptitle("Frame: %d, Cluster: %d" % (i, j))
        ax.imshow(img)
        plt.show()
