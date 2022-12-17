import sys
sys.path.append('../')
import torch.nn as nn
from WGAN.ResNet import *

class Generator_28(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator_28, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            nn.ConvTranspose2d(num_input, 1024, kernel_size=(4,4), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            *Conv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, num_output, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
        )
        print("Generator_28")

    def forward(self, input):
        output = self.Net(input)
        return output

class Generator_32(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator_32, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            nn.ConvTranspose2d(num_input, 1024, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            *Conv(1024, 512),
            *Conv(512, 256),
            nn.ConvTranspose2d(256, num_output, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
            nn.Tanh()
        )
        print("Generator_32")

    def forward(self, input):
        output = self.Net(input)
        return output

class Generator_Res(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator_Res, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(num_input, 1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(True)
        )
        self.Net = nn.Sequential(
            ResNet(1024, 1024),
            # 1024 * 4 * 4
            ResBlockGenerator(1024, 512, kernel_size=(4,4), stride=(2,2), padding=(1,1), activation=nn.ReLU(True)),
            # 64 * 8 * 8
            ResBlockGenerator(512, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1), activation=nn.ReLU(True)),
            # 64 * 16 * 16
            ResBlockGenerator(256, 64, kernel_size=(4,4), stride=(2,2), padding=(1,1), activation=nn.ReLU(True)),
            # 64 * 32 * 32
            nn.Conv2d(64, num_output, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.Tanh()
        )
        print("Generator_Res")

    def forward(self, input):
        input = input.squeeze(-1)
        input = input.squeeze(-1)
        output = self.dense(input)
        output = output.view(-1, 1024, 4, 4)
        output = self.Net(output)
        return output
