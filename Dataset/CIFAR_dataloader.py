import sys
sys.path.append('../')
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np

# Define the train and test sets
dataset_train = CIFAR10(root="../Dataset/", download=True, train=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                           ]))

dataset_test = CIFAR10(root="../Dataset/", download=True, train=False,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                           ]))

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.247, 1/0.243, 1/0.261 ]),
                                transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

batch_size = 64
num_workers = 8
# The loaders perform the actual work
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                         shuffle=True, num_workers= num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                         shuffle=True, num_workers= num_workers, pin_memory=True)
if __name__ == '__main__':
    real_batch = next(iter(train_loader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()