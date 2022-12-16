import sys
sys.path.append('./')
import torch
import os
from torch.autograd import Variable
from torch import autograd
from Data_loader import train_loader_cifar, train_loader_mnist, train_loader_fashionmnist
from torchvision import utils
from WGAN.Generator import Generator_Res, Generator_28, Generator_32
from WGAN.Discriminator import Discriminator_Res, Discriminator_wgan_28, Discriminator_wgan_32, \
    Discriminator_SN_28, Discriminator_SN_32, Discriminator_SN_Res
from WGAN.get_fid_score import get_fid
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class WGAN():
    def __init__(self, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='MNIST', iter=0, G_iter = int(1e4), D_iter = int(5)):
        self.ResNet = ResNet
        self.epoch = 0
        self.maxepochs = int(1e3)
        self.G_losses = []
        self.Real_losses = []
        self.Fake_losses = []
        self.img_list = []
        self.weight_cliping_limit = 0.01
        self.D_iter = D_iter
        self.fid_score = []
        self.best_fid = 1e10
        self.lambda_term = 10
        self.gradient_penalty = gradient_penalty
        self.spectral_norm = spectral_norm
        self.train_set = train_set
        self.iter = str(int(iter))
        self.path = 'WGAN'
        self.generator_iters = G_iter
        if train_set == 'CIFAR':
            self.img_size = 32
            self.output_ch = 3
        else:
            self.img_size = 28
            self.output_ch = 1
        if ResNet:
            self.path += '_ResNet'
            self.G = Generator_Res(100, self.output_ch).to(device)
            self.G_best = Generator_Res(100, self.output_ch).to(device)
            if spectral_norm:
                self.path = 'SN_' + self.path
                self.D = Discriminator_SN_Res(self.output_ch).to(device)
            else:
                self.D = Discriminator_Res(self.output_ch).to(device)
        else:
            if self.img_size == 28:
                self.G = Generator_28(100, self.output_ch).to(device)
                self.G_best = Generator_28(100, self.output_ch).to(device)
                if spectral_norm:
                    self.path = 'SN_' + self.path
                    self.D = Discriminator_SN_28(self.output_ch).to(device)
                else:
                    self.D = Discriminator_wgan_28(self.output_ch).to(device)
            else:
                self.G = Generator_32(100, self.output_ch).to(device)
                self.G_best = Generator_32(100, self.output_ch).to(device)
                if spectral_norm:
                    self.path = 'SN_' + self.path
                    self.D = Discriminator_SN_32(self.output_ch).to(device)
                else:
                    self.D = Discriminator_wgan_32(self.output_ch).to(device)
            if self.gradient_penalty:
                self.path += '_GP'
        self.path += '_' + train_set + '_' + self.iter + '/'
        self.checkpoint = 'checkpoint/'
        self.optim_G = torch.optim.RMSprop(self.G.parameters(), lr=5e-5)
        self.optim_D = torch.optim.RMSprop(self.D.parameters(), lr=5e-5)
        print(self.D)
        print(self.G)

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(real_images.size(0), 1, 1, 1).uniform_(0, 1).to(device)
        eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3))

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).to(device),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def train(self, train_loader):
        print(self.path)
        try:
            os.mkdir(self.checkpoint + self.path)
        except:
            pass
        try:
            self.load()
        except:
            pass
        self.G.train()
        self.D.train()
        self.data = self.get_infinite_batches(train_loader)
        while self.epoch < self.generator_iters+1:
            images = self.data.__next__()
            x = Variable(images).to(device)
            batch_size = x.size(0)
            for p in self.D.parameters():
                p.requires_grad = True
            for i in range(self.D_iter):
                # train the discreiminator
                self.D.zero_grad()
                if not self.spectral_norm:
                    if not self.gradient_penalty:
                        for p in self.D.parameters():
                            p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                D_real = self.D(x)
                loss_real = -D_real.mean(0).view(1)
                loss_real.backward()
                z = Variable(torch.randn((batch_size, 100, 1, 1))).to(device)
                x_fake = self.G(z).detach()
                loss_fake = self.D(x_fake)
                loss_fake = loss_fake.mean(0).view(1)
                loss_fake.backward()
                if not self.spectral_norm:
                    if self.gradient_penalty:
                        gp = self.calculate_gradient_penalty(x.data, x_fake.data)
                        gp.backward()
                self.optim_D.step()
                loss_D = loss_fake + loss_real
                self.Real_losses.append(loss_real.item())
                self.Fake_losses.append(loss_fake.item())
                print("D_real_loss:{}, D_fake_loss:{}".format(loss_real.cpu().detach().numpy(),
                                                          loss_fake.cpu().detach().numpy()))

            z = Variable(torch.randn((batch_size, 100, 1, 1))).to(device)
            self.G.zero_grad()
            for p in self.D.parameters():
                p.requires_grad = False
            x_fake = self.G(z)
            loss_G = self.D(x_fake)
            loss_G = -loss_G.mean(0).view(1)
            # train the generator
            loss_G.backward()
            self.optim_G.step()
            self.G_losses.append(loss_G.item())
            print("epoch:{}/{}, G_loss:{}".format(self.epoch, self.generator_iters, loss_G.cpu().detach().numpy()))

            if self.epoch % 500 == 0:
                self.save()
                self.evaluate()
                fid_score = get_fid(x, x_fake.detach())
                self.fid_score.append(fid_score)
                if fid_score < self.best_fid:
                    self.best_fid = fid_score
                    self.G_best = self.G
                print("FID score: {}".format(fid_score))
            self.epoch += 1

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def save(self):
        torch.save({"epoch": self.epoch,
                    "G_state_dict": self.G.state_dict(),
                    "G_best_state_dict": self.G_best.state_dict(),
                    "optimizer_G": self.optim_G.state_dict(),
                    "losses_G": self.G_losses,
                    "FID scores": self.fid_score,
                    "Best FID score": self.best_fid}, self.checkpoint + self.path + "G.pth")
        torch.save({"D_state_dict": self.D.state_dict(),
                    "optimizer_D": self.optim_D.state_dict(),
                    "losses_fake": self.Fake_losses,
                    "losses_real": self.Real_losses}, self.checkpoint + self.path + "D.pth")
        if self.epoch == self.generator_iters:
            torch.save({"epoch": self.epoch,
                        "G_state_dict": self.G.state_dict(),
                        "G_best_state_dict": self.G_best.state_dict(),
                        "optimizer_G": self.optim_G.state_dict(),
                        "losses_G": self.G_losses,
                        "FID scores": self.fid_score,
                        "Best FID score": self.best_fid}, self.checkpoint + self.path + "G_{}.pth".format(self.epoch))
            torch.save({"D_state_dict": self.D.state_dict(),
                        "optimizer_D": self.optim_D.state_dict(),
                        "losses_fake": self.Fake_losses,
                        "losses_real": self.Real_losses}, self.checkpoint + self.path + "D_{}.pth".format(self.epoch))
        print("model saved! path: " + self.path)

    def load(self):
        checkpoint_G = torch.load(self.checkpoint + self.path + "G.pth")
        checkpoint_D = torch.load(self.checkpoint + self.path + "D.pth")
        self.epoch = checkpoint_G["epoch"]
        self.G.load_state_dict(checkpoint_G["G_state_dict"])
        self.G_best.load_state_dict(checkpoint_G["G_best_state_dict"])
        self.optim_G.load_state_dict(checkpoint_G["optimizer_G"])
        self.G_losses = checkpoint_G["losses_G"]
        self.fid_score = checkpoint_G["FID scores"]
        self.best_fid = checkpoint_G["Best FID score"]
        self.D.load_state_dict(checkpoint_D["D_state_dict"])
        self.optim_D.load_state_dict(checkpoint_D["optimizer_D"])
        self.Fake_losses = checkpoint_D["losses_fake"]
        self.Real_losses = checkpoint_D["losses_real"]
        print("model loaded! path: " + self.path)

    def evaluate(self):
        self.load()
        z = torch.randn((800, 100, 1, 1)).to(device)
        if self.spectral_norm:
            root = 'SN_WGAN'
        elif self.gradient_penalty:
            root = 'WGAN_GP'
        else:
            root = 'WGAN'
        if self.ResNet:
            root += "_Res"
        if self.train_set == "CIFAR":
            path = root + '_CIFAR/'
        elif self.train_set == "MNIST":
            path = root + '_MNIST/'
        else:
            path = root + '_FashionMNIST/'
        try:
            os.mkdir('Results/' + path)
        except:
            pass
        with torch.no_grad():
            fake_img = self.G(z).detach()
            fake_img = fake_img.data.cpu()
            # if self.train_set == "CIFAR":
            #     fake_img = self.invTrans(fake_img)
            # else:
            #     fake_img = fake_img.mul(0.5).add(0.5)
            grid = utils.make_grid(fake_img[:64], normalize=True)
            self.img_list.append(grid)
            torch.save({"img_list": self.img_list}, self.checkpoint + self.path + "img_list.pth".format(self.epoch))
            # utils.save_image(grid, 'Results/' + path + 'img_generatori_iter_{}.png'.format(self.epoch))

    def generate_samples(self):
        if self.spectral_norm:
            root = 'SN_WGAN'
        elif self.gradient_penalty:
            root = 'WGAN_GP'
        else:
            root = 'WGAN'
        if self.ResNet:
            root += "_Res"
        if self.train_set == "CIFAR":
            path = root + '_CIFAR'
        elif self.train_set == "MNIST":
            path = root + '_MNIST'
        else:
            path = root + '_FashionMNIST'
        path += '_' + self.iter + '/'
        try:
            os.mkdir('Results/'+path)
        except:
            pass
        self.load()
        z = torch.randn((800, 100, 1, 1)).to(device)
        with torch.no_grad():
            fake_img = self.G(z).detach()
            fake_img_best = self.G_best(z)
            fake_img = fake_img.data.cpu()
            fake_img_best = fake_img_best.data.cpu()
            # if self.train_set == "CIFAR":
            #     fake_img = self.invTrans(fake_img)
            #     fake_img_best = self.invTrans(fake_img_best)
            # else:
            #     fake_img = fake_img.mul(0.5).add(0.5)
            #     fake_img_best = fake_img_best.mul(0.5).add(0.5)
            plt.show()
            grid = utils.make_grid(fake_img[:64], normalize=True)
            grid_best = utils.make_grid(fake_img_best[:64], normalize=True)
            utils.save_image(grid, 'Results/'+path+'img.png')
            utils.save_image(grid_best, 'Results/'+path+'img_best.png')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="WGAN", choices=['WGAN', 'SN_WGAN', 'WGAN_GP'])
    parser.add_argument('-d', '--dataset', type=str, default="CIFAR", choices=['MNIST', 'CIFAR', 'FashionMNIST'])
    parser.add_argument('-r', '--resnet', type=bool, default=False)
    parser.add_argument('-i', '--iter', type=int, default=1)
    parser.add_argument('-G', '--g_iter', type=int, default=int(1e4))
    parser.add_argument('-D', '--d_iter', type=int, default=int(5))
    args = parser.parse_args()
    gradient_penalty = False
    spectral_norm = False
    if args.dataset == "CIFAR":
        train_loader = train_loader_cifar
    elif args.dataset == "MNIST":
        train_loader = train_loader_mnist
    else:
        train_loader = train_loader_fashionmnist
    if args.model == 'WGAN':
        gradient_penalty = False
        spectral_norm = False
    elif args.model == 'SN_WGAN':
        gradient_penalty = False
        spectral_norm = True
    elif args.model == 'WGAN_GP':
        gradient_penalty = True
        spectral_norm = False
    for i in range(args.iter):
        _WGAN = WGAN(ResNet=args.resnet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm, train_set=args.dataset, iter=i,
                     G_iter=args.g_iter, D_iter=args.d_iter)
        _WGAN.train(train_loader)