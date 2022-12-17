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
            layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))))
            layer.append(nn.ReLU(True))
            return layer
        self.conv = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))

        self.Net = nn.Sequential(
            *Conv(input_nums, 256),
            *Conv(256, 512),
            nn.utils.parametrizations.spectral_norm(nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
            nn.ReLU(True),
            # nn.utils.spectral_norm(self.conv),
            # nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))
        )
        self.flatten = nn.Flatten()
        self.dense = nn.utils.spectral_norm(nn.Linear(1024 * 4 * 4, 1))
        print("Discriminator_SN_28")

    def forward(self, input):
        output = self.Net(input)
        output = self.dense(self.flatten(output))
        output = output.view(-1, 1, 1, 1)
        return output

class Discriminator_SN_32(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_SN_32, self).__init__()
        self.conv1 = nn.Conv2d(input_nums, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        self.linear = nn.Linear(1024 * 4 * 4, 1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv4.weight.data, 1.)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)
        self.Net = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(self.conv2),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(self.conv3),
            nn.ReLU(True),
        )
        # self.conv = nn.utils.spectral_norm(self.conv4)
        # self.conv =  nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        self.flatten = nn.Flatten()
        self.dense = nn.utils.spectral_norm(self.linear)
        print("Discriminator_SN_32")
    def forward(self, input):
        output = self.Net(input)
        output = self.dense(self.flatten(output))
        output = output.view(-1, 1, 1, 1)
        # output = self.conv(output)
        return output

class Discriminator_SN_Res(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_SN_Res, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        self.linear = nn.Linear(1024 * 4 * 4, 1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)
        self.Net = nn.Sequential(
            SN_ResBlockDiscriminator(input_nums, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                     activation=nn.ReLU(True)),
            SN_ResBlockDiscriminator(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                     activation=nn.ReLU(True)),
            SN_ResBlockDiscriminator(1024, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                     activation=nn.ReLU(True)),
        )
        # self.conv = nn.utils.spectral_norm(self.conv1)
        self.flatten = nn.Flatten()
        self.dense = nn.utils.spectral_norm(self.linear)
        print("Discriminator_SN_Res")

    def forward(self, input):
        output = self.Net(input)
        output = self.dense(self.flatten(output))
        output = output.view(-1, 1, 1, 1)
        # output = self.conv(output)
        return output