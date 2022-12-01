import sys
sys.path.append('../')
import torch.nn as nn

class Res_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Block, self).__init__()
        self.Conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=0),
            nn.BatchNorm2d(out_channel),
        )
        self.extra = nn.Sequential()
        if in_channel != out_channel:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1)),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.Conv(x)
        x = self.extra(x)
        return out + x

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet, self).__init__()
        self.Conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.Conv_x = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.blk1 = Res_Block(out_channel, out_channel)
        self.blk2 = Res_Block(out_channel, out_channel)
        self.blk3 = Res_Block(out_channel, out_channel)
        self.blk4 = Res_Block(out_channel, out_channel)

    def forward(self, x):
        out = self.Conv(x)
        x = self.Conv_x(x)
        # out = self.blk4(self.blk3(self.blk2(self.blk1(out))))
        # out = self.blk3(self.blk2(self.blk1(out)))
        out = self.blk2(self.blk1(out))
        return out + x

class Res_Block_D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Block_D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.utils.parametrizations.spectral_norm(self.conv2),
        )
        self.extra = nn.Sequential()
        if in_channel != out_channel:
            self.extra = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(self.conv3),
            )

    def forward(self, x):
        out = self.Conv(x)
        x = self.extra(x)
        return out + x

class ResNet_D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet_D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=0)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(True)
        )
        self.Conv_x = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.parametrizations.spectral_norm(self.conv2),
            nn.ReLU(True)
        )
        self.blk1 = Res_Block_D(out_channel, out_channel)
        self.blk2 = Res_Block_D(out_channel, out_channel)
        self.blk3 = Res_Block_D(out_channel, out_channel)
        self.blk4 = Res_Block_D(out_channel, out_channel)

    def forward(self, x):
        out = self.Conv(x)
        x = self.Conv_x(x)
        # out = self.blk4(self.blk3(self.blk2(self.blk1(out))))
        # out = self.blk3(self.blk2(self.blk1(out)))
        out = self.blk2(self.blk1(out))
        return out + x