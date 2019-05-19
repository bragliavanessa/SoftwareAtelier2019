import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
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


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

validation_split = .2
shuffle_dataset = True
random_seed = 42
batch_size = 4
dataset = ImageDataset(csv_file="./dataset.csv",
                       root_dir="./clusters", transform=None)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_load = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=train_sampler)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_load, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))
        running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), './cnn')
