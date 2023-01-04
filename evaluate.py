import os
import pandas as pd
import torch
from Data_loader import *
from WGAN_models import WGAN
from DCGAN_models import DCGAN
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from Dataset.CIFAR_dataloader import test_loader as cifar_test_loader
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# fid = FrechetInceptionDistance().to(device)
fid = FrechetInceptionDistance()
def get_fid(real_images, fake_images):
    '''
        Takes real image batch and generated 'fake' image batch
        Returns FID score, using the pytorch.metrics package
    '''
    # add 2 extra channels for MNIST (as required by InceptionV3
    if real_images.shape[1] != 3:
        real_images = torch.cat([real_images, real_images, real_images], 1)
    if fake_images.shape[1] != 3:
        fake_images = torch.cat([fake_images, fake_images, fake_images], 1)

    # if images not uint8 format, convert them (required format by fid model)
    if real_images.dtype != torch.uint8 or fake_images.dtype != torch.uint8:
        real_images = real_images.type(torch.ByteTensor)
        fake_images = fake_images.type(torch.ByteTensor)

    fid.update(real_images, real=True)  # <--- currently running out of memory here
    fid.update(fake_images, real=False)
    return fid.compute().item()


train_dataset = {
    "CIFAR":train_loader_cifar,
    "MNIST":train_loader_mnist,
    "FashionMNIST":train_loader_fashionmnist,
}

def get_fid_score(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0):
    if dcgan:
        GAN = DCGAN(ResNet=ResNet, train_set=train_set, iter=iter)
        epoch, iter, G_losses, fid_score, best_fid, Fake_losses, Real_losses = GAN.load_results()
    else:
        GAN = WGAN(ResNet=ResNet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm, train_set=train_set, iter=iter)
        epoch, iter, G_losses, fid_score, best_fid, Fake_losses, Real_losses = GAN.load_results()
    print(epoch)
    return np.array(fid_score).mean(), np.min(np.array(fid_score)), fid_score[-1], np.array(fid_score), \
           np.array(G_losses), np.array(Real_losses), np.array(Fake_losses)

def get_fid_scores(model = 'DCGAN', dataset = 'MNIST', ii = 20):
    fid_scores_mean = []
    fid_scores_min = []
    fid_scores_last = []
    fid_scores = []
    G_losses = []
    Real_losses = []
    Fake_losses = []
    dcgan = False
    spectral_norm = False
    if model == 'DCGAN':
        dcgan = True
    if model == 'SN_WGAN':
        spectral_norm = True
    for i in range(ii):
        try:
            fid_score_mean, fid_score_min, fid_score_last, fid_score, G_loss, Real_loss, Fake_loss= get_fid_score(dcgan=dcgan, ResNet=False, gradient_penalty=False,
                                                          spectral_norm=spectral_norm, train_set=dataset, iter=i)
            fid_scores_mean.append(fid_score_mean)
            fid_scores_min.append(fid_score_min)
            fid_scores_last.append(fid_score_last)
            fid_scores.append(fid_score[:50])
            G_losses.append(G_loss)
            Real_losses.append(Real_loss)
            Fake_losses.append(Fake_loss)
        except:
            pass
    fid_scores_mean = np.array(fid_scores_mean)
    fid_scores_min = np.array(fid_scores_min)
    fid_scores_last = np.array(fid_scores_last)
    fid_scores = np.array(fid_scores)
    G_losses = np.array(G_losses)
    Real_losses = np.array(Real_losses)
    Fake_losses = np.array(Fake_losses)
    MKDIR("Figure")
    MKDIR("Figure/"+model+"/")
    MKDIR("Figure/"+model+"/"+dataset+"/")
    np.save("Figure/"+model+"/"+dataset+"/fid_scores_mean.npy", fid_scores_mean)
    np.save("Figure/"+model+"/"+dataset+"/fid_scores_min.npy", fid_scores_min)
    np.save("Figure/" + model + "/" + dataset + "/fid_scores_last.npy", fid_scores_last)
    np.save("Figure/" + model + "/" + dataset + "/fid_scores.npy", fid_scores)
    np.save("Figure/" + model + "/" + dataset + "/G_losses.npy", G_losses)
    np.save("Figure/" + model + "/" + dataset + "/Real_losses.npy", Real_losses)
    np.save("Figure/" + model + "/" + dataset + "/Fake_losses.npy", Fake_losses)

def save_results():
    models = ['DCGAN', 'WGAN', 'SN_WGAN']
    train_sets = ['CIFAR', 'MNIST', 'FashionMNIST']
    for model in models:
        for train_set in train_sets:
            get_fid_scores(model, train_set)

def show_fid_score(train_set='CIFAR', iter=40000):
    ii = int(iter / 200)
    path = "Figure/"+'SN_WGAN'+"/"+train_set+"/"+"/fid_scores.npy"
    snwgan_fid = np.load(path, allow_pickle=True)
    snwgan_fid = snwgan_fid.mean(axis=0)[:ii]
    path = "Figure/" + 'WGAN' + "/" + train_set + "/" + "/fid_scores.npy"
    wgan_fid = np.load(path, allow_pickle=True)
    wgan_fid = wgan_fid.mean(axis=0)[:ii]
    path = "Figure/" + 'DCGAN' + "/" + train_set + "/" + "/fid_scores.npy"
    dcgan_fid = np.load(path, allow_pickle=True)
    dcgan_fid = dcgan_fid.mean(axis=0)[:ii]
    x1 = np.linspace(0, iter, ii)
    y1 = snwgan_fid
    l1 = plt.plot(x1, y1, 'b--', label='Spectral Normalization WGAN')
    x2 = np.linspace(0, iter, ii)
    y2 = wgan_fid
    l2 = plt.plot(x2, y2, 'g--', label='WGAN')
    x3 = np.linspace(0, iter, ii)
    y3 = dcgan_fid
    l3 = plt.plot(x3, y3, 'r--', label='DCGAN')
    plt.title('FID scores')
    plt.xlabel('iteration')
    plt.ylabel('fid score')
    plt.legend()
    plt.show()

def show_Generator_Losses(dcgan = True, gradient_penalty=False, spectral_norm=False, train_set='CIFAR', iter=0):
    title = ''
    path = "Figure/"
    if dcgan:
        title += "DCGAN "
        model = "DCGAN/"
    else:
        title += "WGAN "
        model = "WGAN/"
    if spectral_norm:
        title = "Spectral Norm " + title
        model = "SN_" + model
    if gradient_penalty:
        title = "Gradient Penalty " + title
    path += model + train_set + '/' + "G_losses.npy"
    G_losses = np.load(path, allow_pickle=True)[iter]
    if train_set == "CIFAR":
        iter = 40000
    else:
        iter = 10000
    # Generator loss
    x1 = np.linspace(0, iter, G_losses.shape[0])
    y1 = np.array(G_losses)
    g_loss = plt.plot(x1, y1, 'r--', label='Generator Loss')
    plt.title(title + 'Generator Loss')
    plt.xlabel('iteration')
    plt.ylabel('Generator Loss')
    plt.legend()
    plt.show()

def show_Discriminator_Losses(dcgan = False, gradient_penalty=False, spectral_norm=True, train_set='CIFAR', iter=0):
    title = ''
    path = "Figure/"
    if dcgan:
        title += "DCGAN "
        model = "DCGAN/"
    else:
        title += "WGAN "
        model = "WGAN/"
    if spectral_norm:
        title = "Spectral Norm " + title
        model = "SN_" + model
    if gradient_penalty:
        title = "Gradient Penalty " + title
    path += model + train_set + '/'
    Fake_losses = np.load(path + "Fake_losses.npy", allow_pickle=True)[iter]
    Real_losses = np.load(path + "Real_losses.npy", allow_pickle=True)[iter]
    if train_set == "CIFAR":
        iter = 40000
    else:
        iter = 10000
    # Discriminator loss
    x2 = np.linspace(0, iter, Fake_losses.shape[0])
    y2 = np.array(Fake_losses)
    fake_loss = plt.plot(x2, y2, 'b--', label='Fake loss')
    x3 = np.linspace(0, iter, Real_losses.shape[0])
    y3 = np.array(Real_losses)
    real_loss = plt.plot(x3, y3, 'g--', label='Real loss')
    x4 = np.linspace(0, iter, Real_losses.shape[0])
    y4 = np.array(Real_losses) + np.array(Fake_losses)
    real_loss = plt.plot(x4, y4, 'm--', label='Discriminator loss')
    plt.title(title + 'Discriminator Loss')
    plt.xlabel('iteration')
    plt.ylabel('Discriminator Loss')
    plt.legend()
    plt.show()


def MKDIR(path):
    try:
        os.mkdir(path)
    except:
        pass

def get_best_model(train_set='FashionMNIST', iters=20):
    best_fid = 1e8
    which_GAN = "DCGAN"
    fid_dict = {}
    for i in range(iters):
        GAN = DCGAN(ResNet=False, train_set=train_set, iter=i)
        try:
            GAN.load()
            fid_dict["DCGAN_"+train_set+"{}".format(i)] = GAN.best_fid
            if GAN.best_fid < best_fid:
                best_GAN = GAN
                best_fid = GAN.best_fid
                print(best_fid)
                which_GAN = "DCGAN_"+train_set+"{}".format(i)
        except:
            pass
    for i in range(iters):
        GAN = WGAN(ResNet=False, spectral_norm=False, train_set=train_set, iter=i)
        try:
            GAN.load()
            fid_dict["WGAN_"+train_set+"{}".format(i)] = GAN.best_fid
            if GAN.best_fid < best_fid:
                best_GAN = GAN
                best_fid = GAN.best_fid
                print(best_fid)
                which_GAN = "WGAN_"+train_set+"{}".format(i)
        except:
            pass
    for i in range(iters):
        GAN = WGAN(ResNet=False, spectral_norm=True,
                   train_set=train_set, iter=i)
        try:
            GAN.load()
            fid_dict["SN_WGAN_"+train_set+"{}".format(i)] = GAN.best_fid
            if GAN.best_fid < best_fid:
                best_GAN = GAN
                best_fid = GAN.best_fid
                print(best_fid)
                which_GAN = "SN_WGAN_"+train_set+"{}".format(i)
        except:
            pass
    fid_dict["best_model"] = which_GAN
    df = pd.DataFrame(fid_dict, index=[0])
    df.to_csv(train_set + "_fid_results.csv")

    return best_GAN, which_GAN, fid_dict

def get_results():
    d = {}
    root = 'checkpoint/'
    ganname = ['DCGAN_', 'WGAN_', 'WGAN_GP_', 'SN_WGAN_']
    dataset = ['MNIST_', 'FashionMNIST_', 'CIFAR_']
    for data in dataset:
        for gan in ganname:
            gan_ = []
            for i in range(10):
                path = root + gan + data + str(int(i)) + '/'
                checkpoint_G = torch.load(path + "G.pth")
                gan_.append(checkpoint_G["Best FID score"])
            d[data+gan] = gan_
    df = pd.DataFrame(data=d)
    df.to_csv('results_fid.csv')

if __name__ == '__main__':
    show_fid_score(train_set="CIFAR", iter=40000)