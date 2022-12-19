import os
import pandas as pd
from Data_loader import *
from WGAN_models import WGAN
from DCGAN_models import DCGAN
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
    print(GAN.epoch)
    return np.array(GAN.fid_score).mean(), np.min(np.array(GAN.fid_score)), GAN.fid_score[-1], np.array(GAN.fid_score), \
           np.array(GAN.G_losses), np.array(GAN.Real_losses), np.array(GAN.Fake_losses)

def get_fid_scores(model = 'DCGAN', dataset = 'CIFAR', ii = 20):
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
            print(G_loss.shape)
            print(Real_loss.shape)
            print(fid_score.shape)
            fid_scores_mean.append(fid_score_mean)
            fid_scores_min.append(fid_score_min)
            fid_scores_last.append(fid_score_last)
            fid_scores.append(fid_score)
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

def show_fid_score(dcgan = False, ResNet=False, gradient_penalty=False, spectral_norm=False, train_set='CIFAR', iter=0):
    j = 0
    for i in range(2):
        # try:
            _DCGAN = DCGAN(ResNet=ResNet, train_set=train_set, spectral_normal=True, iter=i)
            _DCGAN.load()
            if i == 0:
                dcgan_fid = np.array(_DCGAN.fid_score)
            else:
                dcgan_fid += np.array(_DCGAN.fid_score)
            j += 1
        # except:
        #     pass
    dcgan_fid = dcgan_fid / j
    j = 0
    for i in range(2):
        try:
            _WGAN = WGAN(ResNet=ResNet, gradient_penalty=gradient_penalty, spectral_norm=False, train_set=train_set,
                           iter=i)
            _WGAN.load()
            if i == 0:
                wgan_fid = np.array(_WGAN.fid_score)
            else:
                wgan_fid += np.array(_WGAN.fid_score)
            j += 1
        except:
            pass
    wgan_fid = wgan_fid / j
    j = 0
    for i in range(1):
        # try:
            _SNWGAN = WGAN(ResNet=ResNet, gradient_penalty=gradient_penalty, spectral_norm=True, train_set=train_set, iter=i)
            _SNWGAN.load()
            if i == 0:
                snwgan_fid = np.array(_SNWGAN.fid_score)
            else:
                snwgan_fid += np.array(_SNWGAN.fid_score)
            j += 1
        # except:
        #     pass
    snwgan_fid = snwgan_fid / j
    iter = _WGAN.iter
    print(iter)
    x1 = np.linspace(0, iter, len(_SNWGAN.fid_score))
    # y1 = np.array(_SNWGAN.fid_score)
    y1 = snwgan_fid
    l1 = plt.plot(x1, y1, 'b--', label='SN WGAN')
    x2 = np.linspace(0, iter, len(_WGAN.fid_score))
    # y2 = np.array(_WGAN.fid_score)
    y2 = wgan_fid
    l2 = plt.plot(x2, y2, 'g--', label='WGAN')
    x3 = np.linspace(0, iter, len(_DCGAN.fid_score))
    # y3 = np.array(_DCGAN.fid_score)
    y3 = dcgan_fid
    l3 = plt.plot(x3, y3, 'r--', label='DCGAN')
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
    # x2 = np.linspace(0, epoch, len(GAN.Fake_losses))
    # y2 = np.array(GAN.Fake_losses)
    # fake_loss = plt.plot(x2, y2, 'b--', label='Fake loss')
    # x3 = np.linspace(0, epoch, len(GAN.Real_losses))
    # y3 = np.array(GAN.Real_losses)
    # real_loss = plt.plot(x3, y3, 'g--', label='Real loss')
    # x4 = np.linspace(0, epoch, len(GAN.Real_losses))
    # y4 = np.array(GAN.Real_losses) + np.array(GAN.Fake_losses)
    # real_loss = plt.plot(x4, y4, 'm--', label='Discriminator loss')
    plt.title('SN WGAN Generator Loss')
    plt.xlabel('epoch')
    plt.ylabel('Generator Loss')
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
    train_set = "CIFAR"
    # for i in range(10):
    show_fid_score(dcgan = False, ResNet=False, gradient_penalty=False,
                               spectral_norm=True, train_set=train_set, iter=1)
    # show_Losses(dcgan = False, ResNet=False, gradient_penalty=False,
    #                            spectral_norm=True, train_set=train_set, iter=0)
    #     # fid_scores_mean.append(fid_score_mean)
    #     # fid_scores_min.append(fid_score_min)
    # best_model, which_model, fid_dict = get_best_model(train_set=train_set)
    # print(which_model)
    # print(fid_dict)
    # models = ["WGAN","SN_WGAN","DCGAN"]
    # datasets = ["CIFAR", "MNIST", "FashionMNIST"]
    # for model in models:
    #     for dataset in datasets:
    #         # get_fid_scores(model, dataset)
    #         print(model, dataset)
    #         G_losses = np.load("Figure/"+model+"/"+dataset+"/G_losses.npy", allow_pickle=True)
    #         G_losses_new = []
    #         Real_losses = np.load("Figure/"+model+"/"+dataset+"/Real_losses.npy", allow_pickle=True)
    #         Real_losses_new = []
    #         Fake_losses = np.load("Figure/"+model+"/"+dataset+"/Fake_losses.npy", allow_pickle=True)
    #         Fake_losses_new = []
    #         print(G_losses.shape, Real_losses.shape, Fake_losses.shape)
            # for i in range(G_losses.shape[0]):
            #     try:
            #         G_loss_new = np.reshape(np.array(G_losses[i][:782782]), (1001, -1))
            #         G_loss_new = np.average(G_loss_new, axis=1)
            #         G_losses_new.append(G_loss_new)
            #         if model == 'DCGAN':
            #             Real_loss_new = np.reshape(np.array(Real_losses[i][:782782]), (1001, -1))
            #         else:
            #             Real_loss_new = np.reshape(np.array(Real_losses[i]), (1001, -1))
            #         Real_loss_new = np.average(Real_loss_new, axis=1)
            #         Real_losses_new.append(Real_loss_new)
            #         if model == 'DCGAN':
            #             Fake_loss_new = np.reshape(np.array(Fake_losses[i][:782782]), (1001, -1))
            #         else:
            #             Fake_loss_new = np.reshape(np.array(Fake_losses[i]), (1001, -1))
            #         Fake_loss_new = np.average(Fake_loss_new, axis=1)
            #         Fake_losses_new.append(Fake_loss_new)
            #     except:
            #         pass
            # G_losses_new = np.array(G_losses_new)
            # Real_losses_new = np.array(Real_losses_new)
            # Fake_losses_new = np.array(Fake_losses_new)
            # np.save("Figure/"+model+"/"+dataset+"/G_losses.npy", G_losses_new)
            # np.save("Figure/" + model + "/" + dataset + "/Real_losses.npy", Real_losses_new)
            # np.save("Figure/" + model + "/" + dataset + "/Fake_losses.npy", Fake_losses_new)






















