import sys
sys.path.append('../')
import torch.nn as nn

class ResBlockGenerator(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), activation = nn.ReLU(True)):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.deconv2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                          padding=padding, bias=False)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.deconv1.weight.data, 1.)
        nn.init.xavier_uniform(self.deconv2.weight.data, 1.)
        self.Conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            self.deconv1,
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            self.conv1,
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            self.conv2,
        )
        self.extra = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            self.deconv1
        )

    def forward(self, x):
        out = self.Conv(x)
        x = self.extra(x)
        return out + x

class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), activation = nn.ReLU(True)):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            # ResNet(out_channel, out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
        )
        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel)
        )
        self.activation = activation

    def forward(self, x):
        out = self.Conv(x)
        x = self.extra(x)
        return self.activation(out + x)

class SN_ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), activation = nn.ReLU(True)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv4.weight.data, 1.)
        self.Conv = nn.Sequential(
            nn.ReLU(),
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(),
            nn.utils.parametrizations.spectral_norm(self.conv2),
        )
        self.extra = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv3),
            nn.ReLU(),
            nn.utils.parametrizations.spectral_norm((self.conv4))
        )
        self.activation = activation

    def forward(self, x):
        out = self.Conv(x)
        x = self.extra(x)
        return out + x

class Res_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Block, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
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
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.Conv_x = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
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
        out = self.blk4(self.blk3(self.blk2(self.blk1(out))))
        # out = self.blk3(self.blk2(self.blk1(out)))
        # out = self.blk2(self.blk1(out))
        return out + x

class Res_Block_D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Block_D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Conv = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(True),
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
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Conv = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(True)
        )
        self.Conv_x = nn.Sequential(
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
        out = self.blk4(self.blk3(self.blk2(self.blk1(out))))
        # out = self.blk3(self.blk2(self.blk1(out)))
        # out = self.blk2(self.blk1(out))
        return out + x
