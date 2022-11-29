from Data_loader import *
from WGAN.WGAN_models import WGAN
from DCGAN.DCGAN_models import DCGAN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="DCGAN", choices=['DCGAN', 'WGAN', 'SN_WGAN', 'WGAN_GP'])
parser.add_argument('-d', '--dataset', type=str, default="MNIST", choices=['MNIST', 'CIFAR', 'FashionMNIST'])
parser.add_argument('-r', '--resnet', type=bool, default=False)
parser.add_argument('-i', '--iter', type=int, default=1)
args = parser.parse_args()
if __name__ == '__main__':
    MODEL = args.model
    DATASET = args.dataset
    resnet = args.resnet
    iter = args.iter
    if MODEL == 'DCGAN':
        if DATASET == 'CIFAR':
            for i in range(iter):
                _DCGAN = DCGAN(ResNet=resnet, train_set='CIFAR', iter=i)
                _DCGAN.train(train_loader_cifar)
        elif DATASET == 'MNIST':
            for i in range(iter):
                _DCGAN = DCGAN(ResNet=resnet, train_set='MNIST', iter=i)
                _DCGAN.train(train_loader_mnist)
        else:
            for i in range(iter):
                _DCGAN = DCGAN(ResNet=resnet, train_set='FashionMNIST', iter=i)
                _DCGAN.train(train_loader_fashionmnist)
    elif MODEL == 'WGAN':
        if DATASET == 'CIFAR':
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=False, spectral_norm=False, train_set='CIFAR', iter=0)
                _WGAN.train(train_loader_cifar)
        elif DATASET == 'MNIST':
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=False, spectral_norm=False, train_set='MNIST', iter=0)
                _WGAN.train(train_loader_mnist)
        else:
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0)
                _WGAN.train(train_loader_fashionmnist)
    elif MODEL == 'SN_WGAN':
        if DATASET == 'CIFAR':
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=False, spectral_norm=True, train_set='CIFAR', iter=0)
                _WGAN.train(train_loader_cifar)
        elif DATASET == 'MNIST':
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=False, spectral_norm=True, train_set='MNIST', iter=0)
                _WGAN.train(train_loader_mnist)
        else:
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=False, spectral_norm=True, train_set='FashionMNIST', iter=0)
                _WGAN.train(train_loader_fashionmnist)
    elif MODEL == 'WGAN_GP':
        if DATASET == 'CIFAR':
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=True, spectral_norm=False, train_set='CIFAR', iter=0)
                _WGAN.train(train_loader_cifar)
        elif DATASET == 'MNIST':
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=True, spectral_norm=False, train_set='MNIST', iter=0)
                _WGAN.train(train_loader_mnist)
        else:
            for i in range(iter):
                _WGAN = WGAN(ResNet=resnet, gradient_penalty=True, spectral_norm=False, train_set='FashionMNIST', iter=0)
                _WGAN.train(train_loader_fashionmnist)