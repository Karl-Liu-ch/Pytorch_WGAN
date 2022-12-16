from Data_loader import *
from WGAN_models import WGAN
from DCGAN_models import DCGAN

if __name__ == '__main__':
    train_set = "MNIST"
    for i in range(20):
        try:
            _DCGAN = DCGAN(ResNet=False, train_set=train_set, iter=i)
            _DCGAN.generate_samples()
        except:
            pass
    for i in range(20):
        try:
            _WGAN = WGAN(ResNet=False, gradient_penalty=False, spectral_norm=False, train_set=train_set, iter=i)
            _WGAN.generate_samples()
        except:
            pass
    for i in range(20):
        try:
            _WGAN = WGAN(ResNet=False, gradient_penalty=False, spectral_norm=True, train_set=train_set, iter=i)
            _WGAN.generate_samples()
        except:
            pass