import sys
sys.path.append('../')
import torch
import torch.nn as nn
from DCGAN.ResNet import *

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
            ResBlockDiscriminator(input_nums, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), activation=nn.ReLU(True)),
            ResBlockDiscriminator(64, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), activation=nn.ReLU(True)),
            ResBlockDiscriminator(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), activation=nn.ReLU(True)),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.Sigmoid()
        )
        print("Discriminator_Res")

    def forward(self, input):
        output = self.Net(input)
        output = torch.squeeze(output, dim=-1)
        output = torch.squeeze(output, dim=-1)
        return output

class Discriminator_32(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_32, self).__init__()

        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.LeakyReLU(0.2, inplace=True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 64),
            *Conv(64, 256),
            *Conv(256, 512),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.Sigmoid()
        )
        print("Discriminator_32")

    def forward(self, input):
        output = self.Net(input)
        output = torch.squeeze(output, dim=-1)
        output = torch.squeeze(output, dim=-1)
        return output

class Discriminator_28(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_28, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.LeakyReLU(0.2, inplace=True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 64),
            *Conv(64, 256),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid()
        )
        print("Discriminator_28")

    def forward(self, input):
        output = self.Net(input)
        output = torch.squeeze(output, dim=-1)
        output = torch.squeeze(output, dim=-1)
        return output


# class Discriminator_SN_28(nn.Module):
#     def __init__(self, input_nums):
#         super(Discriminator_SN_28, self).__init__()
#         def Conv(input_nums, output_nums):
#             layer = []
#             layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))))
#             layer.append(nn.ReLU(True))
#             return layer
#
#         self.Net = nn.Sequential(
#             *Conv(input_nums, 256),
#             *Conv(256, 512),
#             nn.utils.parametrizations.spectral_norm(nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
#             nn.ReLU(True),
#             nn.utils.parametrizations.spectral_norm(nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))),
#             nn.Sigmoid()
#         )
#         print("Discriminator_wgan_28")
#
#     def forward(self, input):
#         output = self.Net(input)
#         output = torch.squeeze(output, dim=-1)
#         output = torch.squeeze(output, dim=-1)
#         return output

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
            nn.utils.parametrizations.spectral_norm(self.conv),
            nn.Sigmoid()
            # nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))
        )
        print("Discriminator_SN_28")

    def forward(self, input):
        output = self.Net(input)
        output = torch.squeeze(output, dim=-1)
        output = torch.squeeze(output, dim=-1)
        return output

# class Discriminator_SN_32(nn.Module):
#     def __init__(self, input_nums):
#         super(Discriminator_SN_32, self).__init__()
#         def Conv(input_nums, output_nums):
#             layer = []
#             layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))))
#             layer.append(nn.ReLU(True))
#             return layer
#
#         self.Net = nn.Sequential(
#             *Conv(input_nums, 256),
#             *Conv(256, 512),
#             *Conv(512, 1024),
#             nn.utils.parametrizations.spectral_norm(nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=0)),
#             nn.Sigmoid()
#         )
#         print("Discriminator_SN_32")
#
#     def forward(self, input):
#         output = self.Net(input)
#         output = torch.squeeze(output, dim=-1)
#         output = torch.squeeze(output, dim=-1)
#         return output

class Discriminator_SN_32(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator_SN_32, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))))
            layer.append(nn.ReLU())
            layer.append(nn.utils.parametrizations.spectral_norm(nn.Conv2d(output_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))))
            layer.append(nn.ReLU())
            return layer
        self.Net = nn.Sequential(
            *Conv(input_nums, 256),
            *Conv(256, 512),
            *Conv(512, 1024),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))),
            nn.Sigmoid()
        )
        # self.conv =  nn.Conv2d(1024, 1, kernel_size=(4, 4), stride=(1,1), padding=0)
        print("Discriminator_SN_32")
    def forward(self, input):
        output = self.Net(input)
        output = torch.squeeze(output, dim=-1)
        output = torch.squeeze(output, dim=-1)
        return output