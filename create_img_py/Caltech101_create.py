# https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
#https://github.com/kboseong/RotNet

import torch.optim as optim
import os
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.ToTensor(),])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
trainset = torchvision.datasets.STL10(root='./Caltech101/',    #데이터 경로
                          transform=transforms.ToTensor(),
                          # general image format (1~255) H,W,C => pytorch image (0~1) C,H,W
                          download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.STL10(root='./Caltech101/',
                         transform=transforms.ToTensor(),
                         download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)


print_rotation = 3  #rotation number
train_path = "./Caltech101/train/"
if not os.path.exists(train_path):
    os.makedirs(train_path)

for batch_idx, (img, groundTruth) in enumerate(trainloader):
    img1 = img[0]
    file_name = train_path + str(batch_idx) + ".png"
    save_image(img1, file_name)

test_path = "./Caltech101/train/"
if not os.path.exists(test_path):
    os.makedirs(test_path)

for batch_idx, (img, groundTruth) in enumerate(testloader):
    img1 = img[0]
    file_name = test_path + str(batch_idx) + ".png"
    save_image(img1, file_name)