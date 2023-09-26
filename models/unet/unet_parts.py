""" Parts of the U-Net model """

# Copied from:
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import __add__

def padding_size(kernel_sizes):
    # https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121
    return reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]])


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, up_in_channels=None):
        super().__init__()
        if up_in_channels is None: up_in_channels = in_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(up_in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

# Resnet Blocks for Unets
# https://medium.com/@nishanksingla/unet-with-resblock-for-semantic-segmentation-dd1766b4ff66
# https://github.com/Nishanksingla/UNet-with-ResBlock/blob/master/resnet34_unet_model.py

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=True):
        super().__init__()
        self.skip = nn.Sequential(
            nn.ZeroPad2d(padding_size((1, 1))),
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False))
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(padding_size((3, 3))),
            nn.Conv2d(in_channels, out_channels, (3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(padding_size((3, 3))),
            nn.Conv2d(out_channels, out_channels, (3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.max_pool = max_pool
        
    def forward(self, x):
        skip = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + skip
        out = F.relu(out)
        if self.max_pool:
            return out, F.max_pool2d(out, (2, 2))
        else:
            return out


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Sequential(
            nn.ZeroPad2d(padding_size((1, 1))),
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False))
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), stride=2)
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(padding_size((3, 3))),
            nn.Conv2d(in_channels, out_channels, (3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(padding_size((3, 3))),
            nn.Conv2d(out_channels, out_channels, (3, 3), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x1, x2):
        up_sampled = self.up_conv(x1)
        concated = torch.cat((up_sampled, x2), dim=1)
        skip = self.skip(concated)
        out = self.conv1(concated)
        out = self.conv2(out)
        out = out + skip
        out = F.relu(out)
        return out
