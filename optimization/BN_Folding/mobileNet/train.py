import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as transforms
from dataset import *
import numpy as np
from model import *
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import datetime


def train_dataloader():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([256,256]),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 200

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    return trainloader

def test_dataloader():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize([256, 256]),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 200

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)


    return testloader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainloader = train_dataloader()
    testloader  = test_dataloader()

    dataiter = iter(trainloader)
    images, labels = dataiter.next()


    img_grid = torchvision.utils.make_grid(images) #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MobileNet().to(device)
    CEE = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    writer = SummaryWriter(logdir='check_1')  # this code decide tensorboard's path
    writer.add_image('four_fashion_mnist_images', img_grid)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    i = 1
    correct = 0
    total = 0
    model.train()
    for j in range(200):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = inputs.to(device), labels.to(device)
            #print(f"data : {data}, target : {target}")

            optimizer.zero_grad()
            output = model(inputs)

            # acc = accuracy(output, labels,topk=(1,))
            # print(acc.shape)
            loss = CEE(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = correct / total


            print(f"{j+1} - {i+1} -  loss : {running_loss / 2000:.5f}  acc : {acc:.4f}")
            running_loss = 0.0
            writer.add_scalar("Loss/train", loss, i)






    writer.flush()
    writer.close()


if __name__=='__main__':
    main()