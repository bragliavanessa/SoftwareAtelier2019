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
classes_index = torch.arange(0, 3)
classes_one_hot = torch.nn.functional.one_hot(classes_index)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

validation_split = .2
shuffle_dataset = True
random_seed = 42
batch_size = 32
dataset = ImageDataset(csv_file="./dataset.csv",
                       root_dir="./../clusters", transform=None)
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
test_load = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        sampler=valid_sampler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = Net()
net.load_state_dict(torch.load("./cnn"))
net.eval()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))
with torch.no_grad():
    for i, data in enumerate(test_load):
        print("%d/%d" % (i, len(test_load)))
        images, labels = data
        images, labels = images.float().to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(3):
    print('Accuracy of %5s : %2d %% (%d)' % (
        classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))
