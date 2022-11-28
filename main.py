from Data_loader import *
from WGAN.WGAN_models import WGAN
from DCGAN.DCGAN_models import DCGAN
# train wgan on cifar
for i in range(1):
    _WGAN = WGAN(ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='CIFAR', iter=i)
    _WGAN.train(train_loader_cifar)
# train sn_wgan on cifar
for i in range(1):
    _WGAN = WGAN(ResNet=False, gradient_penalty=False, spectral_norm=True, train_set='CIFAR', iter=i)
    _WGAN.train(train_loader_cifar)
# train DCGAN on cifar
for i in range(1):
    _DCGAN = DCGAN(ResNet=False, train_set='CIFAR', iter=i)
    _DCGAN.train(train_loader_cifar)
# train wgan on FashionMNIST
for i in range(3):
    _WGAN = WGAN(ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=i)
    _WGAN.train(train_loader_fashionmnist)
# train sn_wgan on FashionMNIST
for i in range(3):
    _WGAN = WGAN(ResNet=False, gradient_penalty=False, spectral_norm=True, train_set='FashionMNIST', iter=i)
    _WGAN.train(train_loader_fashionmnist)