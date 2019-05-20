import os
import os.path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.transform import resize

classes = ("Yellow", "Red", "Noise")
classes_index = torch.arange(0, 3)
classes_one_hot = torch.nn.functional.one_hot(classes_index)


class ImageDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cluster_frame = pd.read_csv(csv_file, header=None, sep="\t")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.cluster_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.cluster_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = resize(image, (32, 32))
        image = image.transpose((2, 0, 1))
        if self.transform:
            image = self.transform(image)
        cluster = self.cluster_frame.iloc[idx, 1:].values[0]
        index = classes.index(cluster)
        # one_hot = classes_one_hot[index]
        sample = (image, index)

        return sample
