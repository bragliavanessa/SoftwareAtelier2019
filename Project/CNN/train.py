import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from Model import Net
from Dataset import ImageDataset

classes = ("Yellow", "Red", "Noise")

validation_split = .1
shuffle_dataset = True
random_seed = 42
batch_size = 16
dataset = ImageDataset(csv_file="./dataset.csv",
                       root_dir="./../clusters")
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.7)

# Training
for epoch in range(5):
    for i, data in enumerate(train_load, 0):
        inputs, labels, name = data
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        print('[%d, %5d/%d] loss: %.3f' %
              (epoch + 1, i + 1, len(train_load), loss.item()))

print('Finished Training')

torch.save(net.state_dict(), './cnn')
