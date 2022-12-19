import sys
sys.path.append('./')
from DCGAN.Generator import *
from DCGAN.Discriminator import *
from torch.autograd import Variable
from Data_loader import train_loader_cifar, train_loader_mnist, train_loader_fashionmnist
from DCGAN.get_fid_score import get_fid
import os
from torchvision import utils
import torchvision.transforms as transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGAN():
    def __init__(self, ResNet = False, train_set = "CIFAR", spectral_normal = False, iter = 0, G_iter = int(4e4), D_iter = int(5)):
        self.ResNet = ResNet
        self.path_iter = str(int(iter))
        self.path = 'DCGAN'
        self.spectral_normal = spectral_normal
        if spectral_normal:
            self.path = 'SN_' + self.path
        self.epoch = 0
        self.maxepochs = int(1e3)
        self.loss_func = nn.BCELoss()
        self.G_losses = []
        self.Real_losses = []
        self.Fake_losses = []
        self.fid_score = []
        self.best_fid = 1e10
        self.train_set = train_set
        self.generator_iters = G_iter
        self.D_iter = D_iter
        self.iter = 0
        if train_set == "CIFAR":
            self.output_ch = 3
            if ResNet:
                self.path += "_" + "Res"
                self.G = Generator_Res(100, self.output_ch).to(device)
                self.G_best = Generator_Res(100, 3).to(device)
                self.D = Discriminator_Res(self.output_ch).to(device)
            elif spectral_normal:
                self.G = Generator_32(100, self.output_ch).to(device)
                self.G_best = Generator_32(100, self.output_ch).to(device)
                self.D = Discriminator_SN_32(self.output_ch).to(device)
            else:
                self.G = Generator_32(100, self.output_ch).to(device)
                self.G_best = Generator_32(100, self.output_ch).to(device)
                self.D = Discriminator_32(self.output_ch).to(device)
        else:
            self.output_ch = 1
            if spectral_normal:
                # self.G = Generator_28(100, self.output_ch).to(device)
                # self.G_best = Generator_28(100, self.output_ch).to(device)
                # self.D = Discriminator_SN_28(self.output_ch).to(device)
                self.G = Generator_32(100, self.output_ch).to(device)
                self.G_best = Generator_32(100, self.output_ch).to(device)
                self.D = Discriminator_SN_32(self.output_ch).to(device)
            else:
                # self.G = Generator_28(100, self.output_ch).to(device)
                # self.G_best = Generator_28(100, self.output_ch).to(device)
                # self.D = Discriminator_28(self.output_ch).to(device)
                self.G = Generator_32(100, self.output_ch).to(device)
                self.G_best = Generator_32(100, self.output_ch).to(device)
                self.D = Discriminator_32(self.output_ch).to(device)
        self.path += "_" + train_set + '_' + self.path_iter + '/'
        if spectral_norm:
            self.optim_G = torch.optim.RMSprop(self.G.parameters(), lr=5e-5)
            self.optim_D = torch.optim.RMSprop(self.D.parameters(), lr=5e-5)
        else:
            self.optim_G = torch.optim.Adam(self.G.parameters(),lr=1e-4, betas=(0.5, 0.999))
            self.optim_D = torch.optim.Adam(self.D.parameters(),lr=1e-4, betas=(0.5, 0.999))
        self.checkpoint = 'checkpoint/'
        print(self.D)
        print(self.G)

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def train(self, train_loader):
        try:
            os.mkdir(self.checkpoint+self.path)
        except:
            pass
        try:
            self.load()
        except:
            self.D.apply(weights_init)
            print('parameters initialization')
        self.data = self.get_infinite_batches(train_loader)
        # while (self.epoch < self.maxepochs + 1) and (self.iter < self.generator_iters + 1):
        while self.iter < self.generator_iters + 1:
            for i, data in enumerate(train_loader, 0):
                x = Variable(data[0]).to(device)
                batch_size = x.size(0)
                true_label = torch.ones(batch_size, 1).to(device)
                fake_label = torch.zeros(batch_size, 1).to(device)
                self.D.zero_grad()
                self.G.zero_grad()
                D_real = self.D(x)
                loss_real = self.loss_func(D_real, true_label)
                loss_real.backward()
                z = Variable(torch.randn((batch_size, 100, 1, 1))).to(device)
                x_fake = self.G(z)
                D_fake = self.D(x_fake.detach())
                loss_fake = self.loss_func(D_fake, fake_label)
                loss_fake.backward()
                self.optim_D.step()
                self.Real_losses.append(loss_real.item())
                self.Fake_losses.append(loss_fake.item())
                x_fake = self.G(z)
                loss_G = self.D(x_fake)
                loss_G = self.loss_func(loss_G, true_label)
                loss_G.backward()
                self.optim_G.step()
                self.G_losses.append(loss_G.item())
                print("iter: {}/{}, epoch:{}/{}, G_loss:{}".format(self.iter, self.generator_iters, self.epoch, self.maxepochs, loss_G.cpu().detach().numpy()))
                print("D_real_loss:{}, D_fake_loss:{}".format(loss_real.cpu().detach().numpy(),
                                                                       loss_fake.cpu().detach().numpy()))

                if self.iter % 200 == 0:
                    self.save()
                    self.evaluate()
                    fid_score = get_fid(x, x_fake.detach())
                    self.fid_score.append(fid_score)
                    if fid_score < self.best_fid:
                        self.best_fid = fid_score
                        self.G_best = self.G
                    print("FID score: {}".format(fid_score))
                self.iter += 1
            self.epoch += 1

    def save(self):
        torch.save({"epoch": self.epoch,
                    "iter": self.iter,
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
        if (self.epoch == self.generator_iters) or (self.iter == self.generator_iters):
            torch.save({"epoch": self.epoch,
                        "iter": self.iter,
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
        self.G.load_state_dict(checkpoint_G["G_state_dict"])
        self.G_best.load_state_dict(checkpoint_G["G_best_state_dict"])
        self.optim_G.load_state_dict(checkpoint_G["optimizer_G"])
        self.epoch = checkpoint_G["epoch"]
        self.iter = checkpoint_G["iter"]
        self.G_losses = checkpoint_G["losses_G"]
        self.fid_score = checkpoint_G["FID scores"]
        self.best_fid = checkpoint_G["Best FID score"]
        self.D.load_state_dict(checkpoint_D["D_state_dict"])
        self.optim_D.load_state_dict(checkpoint_D["optimizer_D"])
        self.Fake_losses = checkpoint_D["losses_fake"]
        self.Real_losses = checkpoint_D["losses_real"]
        print("model loaded! path: " + self.path)

    def evaluate(self):
        root = 'DCGAN'
        if self.ResNet:
            root += "_Res"
        if self.spectral_normal:
            root = "SN_" + root
        if self.train_set == "CIFAR":
            path = root + '_CIFAR/'
        elif self.train_set == "MNIST":
            path = root + '_MNIST/'
        else:
            path = root + '_FashionMNIST/'
        try:
            os.mkdir('Results/'+path)
        except:
            pass
        self.load()
        z = torch.randn((800, 100, 1, 1)).to(device)
        with torch.no_grad():
            fake_img = self.G(z)
            fake_img = fake_img.data.cpu()
            grid = utils.make_grid(fake_img[:64], normalize=True)
            utils.save_image(grid, 'Results/'+path+'img_generatori_iter_{}.png'.format(self.iter))

    def generate_samples(self):
        root = 'DCGAN'
        if self.ResNet:
            root += "_Res"
        if self.train_set == "CIFAR":
            path = root + '_CIFAR'
        elif self.train_set == "MNIST":
            path = root + '_MNIST'
        else:
            path = root + '_FashionMNIST'
        path += '_' + str(int(self.iter)) + '/'
        try:
            os.mkdir('Results/'+path)
        except:
            pass
        self.load()
        z = torch.randn((64, 100, 1, 1)).to(device)
        with torch.no_grad():
            fake_img = self.G(z)
            fake_img_best = self.G_best(z)
            fake_img = fake_img.data.cpu()
            fake_img_best = fake_img_best.data.cpu()
            grid = utils.make_grid(fake_img, normalize=True)
            grid_best = utils.make_grid(fake_img_best, normalize=True)
            utils.save_image(grid, 'Results/'+path+'img.png')
            utils.save_image(grid_best, 'Results/'+path+'img_best.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="DCGAN", choices=['DCGAN', 'SN_DCGAN'])
    parser.add_argument('-d', '--dataset', type=str, default="CIFAR", choices=['MNIST', 'CIFAR', 'FashionMNIST'])
    parser.add_argument('-r', '--resnet', type=bool, default=False)
    parser.add_argument('-i', '--iter', type=int, default=1)
    parser.add_argument('-G', '--g_iter', type=int, default=int(1e4))
    parser.add_argument('-D', '--d_iter', type=int, default=int(1))
    args = parser.parse_args()
    gradient_penalty = False
    spectral_norm = False
    if args.dataset == "CIFAR":
        train_loader = train_loader_cifar
    elif args.dataset == "MNIST":
        train_loader = train_loader_mnist
    else:
        train_loader = train_loader_fashionmnist
    if args.model == "SN_DCGAN":
        spectral_norm = True
    for i in range(args.iter):
        _DCGAN = DCGAN(ResNet=args.resnet, train_set=args.dataset, spectral_normal=spectral_norm, iter=i, G_iter=args.g_iter, D_iter=args.d_iter)
        _DCGAN.train(train_loader)