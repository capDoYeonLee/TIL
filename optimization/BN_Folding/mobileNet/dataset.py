import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision.models as models
import matplotlib.pyplot as plt

"""
train_data = torchvision.datasets.CIFAR100('./', train=True, download=True)

# Stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

# calculate the mean and std along the (0, 1) axes
mean = np.mean(x, axis=(0, 1))/255
std = np.std(x, axis=(0, 1))/255
batch_size = 256

# the the mean and std
mean=mean.tolist()
std=std.tolist()

"""


def train_dataloader(x, batch_size=256):

    transform_train = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                                  tt.RandomHorizontalFlip(),
                                  tt.ToTensor(),
                                  tt.Normalize(mean, std, inplace=True)])

    trainset = torchvision.datasets.CIFAR100("./",
                                             train=True,
                                             download=True,
                                             transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    device = get_default_device()
    trainloader = DeviceDataLoader(trainloader, device)

    return trainloader

def test_dataloader(x, batch_size=256):




def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)