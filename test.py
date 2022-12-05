from Data_loader import *
from WGAN_models import WGAN
from DCGAN_models import DCGAN

if __name__ == '__main__':
    # for i in range(3):
    #     _DCGAN = DCGAN(ResNet=False, train_set='CIFAR', iter=i)
    #     _DCGAN.generate_samples()
    # for i in range(3):
    #     _WGAN = WGAN(ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='CIFAR', iter=i)
    #     _WGAN.generate_samples()
    for i in range(3):
        _WGAN = WGAN(ResNet=False, gradient_penalty=False, spectral_norm=True, train_set='CIFAR', iter=i)
        _WGAN.generate_samples()