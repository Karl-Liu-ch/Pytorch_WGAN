from Data_loader import *
from WGAN.WGAN_models import WGAN
from DCGAN.DCGAN_models import DCGAN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

train_dataset = {
    "CIFAR":train_loader_cifar,
    "MNIST":train_loader_mnist,
    "FashionMNIST":train_loader_fashionmnist,
}

def show_fid_score(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0):
    if dcgan:
        GAN = DCGAN(ResNet=ResNet, train_set=train_set, iter=iter)
        GAN.load()
    else:
        GAN = WGAN(ResNet=ResNet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm, train_set='FashionMNIST', iter=iter)
        GAN.load()
    epoch = GAN.epoch
    x = np.linspace(0, epoch, len(GAN.fid_score))
    y = np.array(GAN.fid_score)
    l1 = plt.plot(x, y, 'r--', label='type1')
    # plt.plot(x, y, 'ro-')
    plt.title('FID scores')
    plt.xlabel('epoch')
    plt.ylabel('fid score')
    plt.legend()
    plt.show()

def show_Losses(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0):
    if dcgan:
        GAN = DCGAN(ResNet=ResNet, train_set=train_set, iter=iter)
        GAN.load()
    else:
        GAN = WGAN(ResNet=ResNet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm, train_set='FashionMNIST', iter=iter)
        GAN.load()
    epoch = GAN.epoch
    # x = np.linspace(0, epoch, len(GAN.fid_score))
    # y = np.array(GAN.fid_score)
    # l1 = plt.plot(x, y, 'r--', label='type1')
    x1 = np.linspace(0, epoch, len(GAN.G_losses))
    y1 = np.array(GAN.G_losses)
    g_loss = plt.plot(x1, y1, 'r--', label='Generator Loss')
    x2 = np.linspace(0, epoch, len(GAN.Fake_losses))
    y2 = np.array(GAN.Fake_losses)
    fake_loss = plt.plot(x2, y2, 'b--', label='Fake loss')
    x3 = np.linspace(0, epoch, len(GAN.Real_losses))
    y3 = np.array(GAN.Real_losses)
    real_loss = plt.plot(x3, y3, 'g--', label='Real loss')
    plt.title('Losses')
    plt.xlabel('epoch')
    plt.ylabel('fid score')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    show_fid_score(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0)
    # show_Losses(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0)