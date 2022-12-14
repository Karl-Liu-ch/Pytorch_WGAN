import sys
sys.path.append('../')
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
    def __init__(self, ResNet = False, train_set = "CIFAR", iter = 0):
        self.ResNet = ResNet
        self.iter = str(int(iter))
        self.path = 'DCGAN'
        self.epoch = 0
        self.maxepochs = int(1e3)
        self.loss_func = nn.BCELoss()
        self.G_losses = []
        self.Real_losses = []
        self.Fake_losses = []
        self.fid_score = []
        self.best_fid = 1e10
        self.train_set = train_set
        if train_set == "CIFAR":
            self.invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                     std=[1 / 0.247, 1 / 0.243, 1 / 0.261]),
                                                transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465],
                                                                     std=[1., 1., 1.]),
                                                ])
            self.output_ch = 3
            if ResNet:
                self.path += "_" + "Res"
                self.G = Generator_Res(100, self.output_ch).to(device)
                self.G_best = Generator_Res(100, 3).to(device)
                self.D = Discriminator_Res(self.output_ch).to(device)
            else:
                self.G = Generator_32(100, self.output_ch).to(device)
                self.G_best = Generator_32(100, self.output_ch).to(device)
                self.D = Discriminator_32(self.output_ch).to(device)
        else:
            self.output_ch = 1
            self.G = Generator_28(100, self.output_ch).to(device)
            self.G_best = Generator_28(100, self.output_ch).to(device)
            self.D = Discriminator_28(self.output_ch).to(device)
        if train_set == "MNIST":
            self.invTrans = transforms.Compose([transforms.Normalize(mean=[0.],
                                                                     std=[1 / 0.1306]),
                                                transforms.Normalize(mean=[-0.4914],
                                                                     std=[1.]),
                                                ])
        if train_set == "FashionMNIST":
            self.invTrans = transforms.Compose([transforms.Normalize(mean=[0.],
                                                                     std=[1 / 0.353]),
                                                transforms.Normalize(mean=[-0.286],
                                                                     std=[1.]),
                                                ])
        self.path += "_" + train_set + '_' + self.iter + '/'
        self.optim_G = torch.optim.Adam(self.G.parameters(),lr=1e-4, betas=(0.5, 0.999))
        self.optim_D = torch.optim.Adam(self.D.parameters(),lr=1e-4, betas=(0.5, 0.999))
        self.checkpoint = 'checkpoint/'
        print(self.D)
        print(self.G)

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
        while self.epoch < self.maxepochs + 1:
            for x, _ in train_loader:
                x = Variable(x).to(device)
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
            print("epoch:{}, G_loss:{}".format(self.epoch, loss_G.cpu().detach().numpy()))
            print("D_real_loss:{}, D_fake_loss:{}".format(loss_real.cpu().detach().numpy(),
                                                                   loss_fake.cpu().detach().numpy()))
            if self.epoch % 20 == 0:
                self.save()
                self.evaluate()
                fid_score = get_fid(x, x_fake.detach())
                self.fid_score.append(fid_score)
                if fid_score < self.best_fid:
                    self.best_fid = fid_score
                    self.G_best = self.G
                print("FID score: {}".format(fid_score))
            self.epoch += 1

    def save(self):
        torch.save({"epoch": self.epoch,
                    "G_state_dict": self.G.state_dict(),
                    "G_best_state_dict": self.G_best.state_dict(),
                    "optimizer_G": self.optim_G.state_dict(),
                    "losses_G": self.G_losses,
                    "FID scores": self.fid_score,
                    "Best FID score": self.best_fid}, self.checkpoint+self.path+"G.pth")
        torch.save({"D_state_dict": self.D.state_dict(),
                    "optimizer_D": self.optim_D.state_dict(),
                    "losses_fake": self.Fake_losses,
                    "losses_real": self.Real_losses}, self.checkpoint+self.path+"D.pth")
        if self.epoch == self.maxepochs:
            torch.save({"epoch": self.epoch,
                        "G_state_dict": self.G.state_dict(),
                        "G_best_state_dict": self.G_best.state_dict(),
                        "optimizer_G": self.optim_G.state_dict(),
                        "losses_G": self.G_losses,
                        "FID scores": self.fid_score,
                        "Best FID score": self.best_fid}, self.checkpoint+self.path+"G_{}.pth".format(self.epoch))
            torch.save({"D_state_dict": self.D.state_dict(),
                        "optimizer_D": self.optim_D.state_dict(),
                        "losses_fake": self.Fake_losses,
                        "losses_real": self.Real_losses}, self.checkpoint+self.path+"D_{}.pth".format(self.epoch))
        print("model saved! path: "+self.path)

    def load(self):
        checkpoint_G = torch.load(self.checkpoint+self.path+"G.pth")
        checkpoint_D = torch.load(self.checkpoint+self.path+"D.pth")
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
        print("model loaded! path: "+self.path)

    def evaluate(self):
        root = 'DCGAN'
        if self.ResNet:
            root += "_Res"
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
        z = torch.randn((1, 100, 1, 1)).to(device)
        with torch.no_grad():
            fake_img = self.G(z)
            fake_img = fake_img.data.cpu()
            if self.train_set == "CIFAR":
                fake_img = self.invTrans(fake_img)
            else:
                fake_img = fake_img.mul(0.5).add(0.5)
            grid = utils.make_grid(fake_img)
            utils.save_image(grid, 'Results/'+path+'img_generatori_iter_{}.png'.format(self.epoch))

if __name__ == '__main__':
    # Train CIFAR on DCGAN 3 times.
    for i in range(3):
        _DCGAN = DCGAN(ResNet = False, train_set = "CIFAR", iter=i)
        _DCGAN.train(train_loader_cifar)
    # Train MNIST on DCGAN 3 times.
    for i in range(3):
        _DCGAN = DCGAN(ResNet=False, train_set="MNIST", iter=i)
        _DCGAN.train(train_loader_mnist)
    # Train FashionMNIST on DCGAN 3 times.
    for i in range(3):
        _DCGAN = DCGAN(ResNet=False, train_set="FashionMNIST", iter=i)
        _DCGAN.train(train_loader_fashionmnist)
    # Train CIFAR on ResNet_DCGAN 1 times.
    for i in range(1):
        _DCGAN = DCGAN(ResNet = True, train_set = "CIFAR", iter=i)
        _DCGAN.train(train_loader_cifar)