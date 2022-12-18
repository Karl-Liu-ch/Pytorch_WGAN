import sys
sys.path.append('../')
import torch.nn as nn
from WGAN.ResNet import *

class Discriminator_wgan_28(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_wgan_28, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 256),
            *Conv(256, 512),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
        )
        print("Discriminator_wgan_28")

    def forward(self, input):
        output = self.Net(input)
        return output

class Discriminator_wgan_32(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_wgan_32, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 256),
            *Conv(256, 512),
            *Conv(512, 1024),
        )
        self.conv = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        print("Discriminator_wgan_32")

    def forward(self, input):
        output = self.Net(input)
        output = self.conv(output)
        return output

class Discriminator_Res(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_Res, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            ResBlockDiscriminator(input_nums, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), activation=nn.ReLU(True)),
            ResBlockDiscriminator(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), activation=nn.ReLU(True)),
            ResBlockDiscriminator(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), activation=nn.ReLU(True)),
        )
        self.conv = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        print("Discriminator_Res")

    def forward(self, input):
        output = self.Net(input)
        output = self.conv(output)
        return output

class Discriminator_SN_28(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_SN_28, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
            layer.append(nn.ReLU(True))
            layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(output_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))))
            layer.append(nn.ReLU(True))
            return layer
        self.conv = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))

        self.Net = nn.Sequential(
            *Conv(input_nums, 256),
            *Conv(256, 512),
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
            nn.ReLU(True),
            nn.utils.spectral_norm(self.conv),
            # nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))
        )
        print("Discriminator_SN_28")

    def forward(self, input):
        output = self.Net(input)
        return output
'''
class Discriminator_SN_32(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_SN_32, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
            layer.append(nn.ReLU(True))
            layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(output_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))))
            layer.append(nn.ReLU(True))
            return layer
        self.Net = nn.Sequential(
            *Conv(input_nums, 256),
            *Conv(256, 512),
            *Conv(512, 1024),
        )
        self.conv = nn.utils.spectral_norm(nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)))
        # self.conv =  nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        print("Discriminator_SN_32")
    def forward(self, input):
        output = self.Net(input)
        output = self.conv(output)
        return output
'''
class Discriminator_SN_Res(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_SN_Res, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        self.Net = nn.Sequential(
            SN_ResBlockDiscriminator(input_nums, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                     activation=nn.ReLU(True)),
            SN_ResBlockDiscriminator(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                     activation=nn.ReLU(True)),
            SN_ResBlockDiscriminator(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                     activation=nn.ReLU(True)),
            nn.ReLU(),
        )
        self.conv = nn.utils.spectral_norm(self.conv1)
        print("Discriminator_SN_Res")

    def forward(self, input):
        output = self.Net(input)
        output = self.conv(output)
        return output
    
leak = 0.1
w_g = 4
class Discriminator_SN_32(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_SN_32, self).__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(input_nums, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = nn.utils.spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))

        self.fc = nn.utils.spectral_norm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))

        return self.fc(m.view(-1,w_g * w_g * 512))