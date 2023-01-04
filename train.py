from Data_loader import *
from WGAN_models import WGAN
from DCGAN_models import DCGAN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="DCGAN", choices=['DCGAN', 'SN_DCGAN', 'WGAN', 'SN_WGAN', 'WGAN_GP', 'SN_WGAN_GP'])
parser.add_argument('-d', '--dataset', type=str, default="CIFAR", choices=['MNIST', 'CIFAR', 'FashionMNIST'])
parser.add_argument('-r', '--resnet', type=bool, default=False)
parser.add_argument('-i', '--iter', type=int, default=1)
parser.add_argument('-G', '--g_iter', type=int, default=int(4e4))
parser.add_argument('-D', '--d_iter', type=int, default=int(1))
args = parser.parse_args()
gradient_penalty = False
spectral_norm = False
if __name__ == '__main__':
    MODEL = args.model
    DATASET = args.dataset
    resnet = args.resnet
    iter = args.iter
    if args.dataset == "CIFAR":
        train_loader = train_loader_cifar
    elif args.dataset == "MNIST":
        train_loader = train_loader_mnist
    else:
        train_loader = train_loader_fashionmnist

    if args.model == "DCGAN":
        for i in range(args.iter):
            _DCGAN = DCGAN(ResNet=args.resnet, train_set=args.dataset, spectral_normal=spectral_norm, iter=i,
                           G_iter=args.g_iter, D_iter=args.d_iter)
            _DCGAN.train(train_loader)
    if args.model == "SN_DCGAN":
        spectral_norm = True
        for i in range(args.iter):
            _DCGAN = DCGAN(ResNet=args.resnet, train_set=args.dataset, spectral_normal=spectral_norm, iter=i,
                           G_iter=args.g_iter, D_iter=args.d_iter)
            _DCGAN.train(train_loader)
    if args.model == 'WGAN':
        gradient_penalty = False
        spectral_norm = False
        for i in range(args.iter):
            _WGAN = WGAN(ResNet=args.resnet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm,
                         train_set=args.dataset, iter=i,
                         G_iter=args.g_iter, D_iter=args.d_iter)
            _WGAN.train(train_loader)
    elif args.model == 'SN_WGAN':
        gradient_penalty = False
        spectral_norm = True
        for i in range(args.iter):
            _WGAN = WGAN(ResNet=args.resnet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm,
                         train_set=args.dataset, iter=i,
                         G_iter=args.g_iter, D_iter=args.d_iter)
            _WGAN.train(train_loader)
    elif args.model == 'WGAN_GP':
        gradient_penalty = True
        spectral_norm = False
        for i in range(args.iter):
            _WGAN = WGAN(ResNet=args.resnet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm,
                         train_set=args.dataset, iter=i,
                         G_iter=args.g_iter, D_iter=args.d_iter)
            _WGAN.train(train_loader)
    elif args.model == 'SN_WGAN_GP':
        gradient_penalty = True
        spectral_norm = True
        for i in range(args.iter):
            _WGAN = WGAN(ResNet=args.resnet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm,
                         train_set=args.dataset, iter=i,
                         G_iter=args.g_iter, D_iter=args.d_iter)
            _WGAN.train(train_loader)