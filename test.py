# load mnist dataset in pytorch
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt

#path = '~/.torch/datasets/mnist'

transform = Compose([ToTensor()])

# download and define the datasets
train = mnist(train=True, download=True, transform=transform)
test = mnist(train=False, download=True, transform=transform)

# define how to enumerate the datasets
train_dl = DataLoader(train, batch_size=32, shuffle=True)
test_dl = DataLoader(test, batch_size=32, shuffle=True)

# get one batch of images
i, (inputs, targets) = next(enumerate(train_dl))

# plot some images
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(inputs[i][0], cmap='gray')

plt.show()
