import os
import pandas as pd
from Data_loader import *
from WGAN.WGAN_models import WGAN
from DCGAN.DCGAN_models import DCGAN
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

train_dataset = {
    "CIFAR":train_loader_cifar,
    "MNIST":train_loader_mnist,
    "FashionMNIST":train_loader_fashionmnist,
}

def get_fid_score(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0):
    if dcgan:
        GAN = DCGAN(ResNet=ResNet, train_set=train_set, iter=iter)
        GAN.load()
    else:
        GAN = WGAN(ResNet=ResNet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm, train_set=train_set, iter=iter)
        GAN.load()
    return np.array(GAN.fid_score).mean(), np.min(np.array(GAN.fid_score))

def get_fid_scores(model = 'DCGAN', dataset = 'CIFAR', ii = 20):
    fid_scores_mean = []
    fid_scores_min = []
    dcgan = False
    spectral_norm = False
    if model == 'DCGAN':
        dcgan = True
    if model == 'SN_WGAN':
        spectral_norm = True
    for i in range(ii):
        try:
            fid_score_mean, fid_score_min = get_fid_score(dcgan=dcgan, ResNet=False, gradient_penalty=False,
                                                          spectral_norm=spectral_norm, train_set=dataset, iter=i)
            fid_scores_mean.append(fid_score_mean)
            fid_scores_min.append(fid_score_min)
        except:
            pass
    fid_scores_mean = np.array(fid_scores_mean)
    fid_scores_min = np.array(fid_scores_min)
    MKDIR("Figure")
    MKDIR("Figure/"+model+"/")
    MKDIR("Figure/"+model+"/"+dataset+"/")
    np.save("Figure/"+model+"/"+dataset+"/fid_scores_mean.npy", fid_scores_mean)
    np.save("Figure/"+model+"/"+dataset+"/fid_scores_min.npy", fid_scores_min)

def show_fid_score(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0):
    if dcgan:
        GAN = DCGAN(ResNet=ResNet, train_set=train_set, iter=iter)
        GAN.load()
    else:
        GAN = WGAN(ResNet=ResNet, gradient_penalty=gradient_penalty, spectral_norm=spectral_norm, train_set=train_set, iter=iter)
        GAN.load()
    epoch = GAN.epoch
    x = np.linspace(0, epoch, len(GAN.fid_score))
    y = np.array(GAN.fid_score)
    l1 = plt.plot(x, y, 'r--', label='type1')
    plt.title('FID scores')
    plt.xlabel('epoch')
    plt.ylabel('fid score')
    plt.legend()
    plt.show()
    return np.array(GAN.fid_score).mean(), np.min(np.array(GAN.fid_score))

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

if __name__ == '__main__':
    # fid_scores_mean = []
    # fid_scores_min = []
    train_set = "FashionMNIST"
    for i in range(10):
        fid_score_mean, fid_score_min = show_fid_score(dcgan = True, ResNet=False, gradient_penalty=False,
                                   spectral_norm=False, train_set=train_set, iter=i)
        # fid_scores_mean.append(fid_score_mean)
        # fid_scores_min.append(fid_score_min)
    best_model, which_model, fid_dict = get_best_model(train_set=train_set)
    print(which_model)
    print(fid_dict)
    # models = ["DCGAN","WGAN","SN_WGAN"]
    # datasets = ["CIFAR", "MNIST", "FashionMNIST"]
    # for model in models:
    #     for dataset in datasets:
    #         get_fid_scores(model, dataset)
    # d = {}
    # fid_scores_mean = np.load("Figure/SN_WGAN/CIFAR/fid_scores_mean.npy")
    # fid_scores_min = np.load("Figure/SN_WGAN/CIFAR/fid_scores_min.npy")
    # sb.boxplot([fid_scores_mean, fid_scores_min])
    # plt.show()
    # show_Losses(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='FashionMNIST', iter=0)