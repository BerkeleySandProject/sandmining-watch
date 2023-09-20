""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNetResBlocks(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = ResBlockDown(self.n_channels, 64)
        self.down2 = ResBlockDown(64, 128)
        self.down3 = ResBlockDown(128, 256)
        self.down4 = ResBlockDown(256, 512)

        self.bottleneck = ResBlockDown(512, 1024, max_pool=False)

        self.up1 = ResBlockUp(1024, 512)
        self.up2 = ResBlockUp(512, 256)
        self.up3 = ResBlockUp(256, 128)
        self.up4 = ResBlockUp(128, 64)

        self.classifier = OutConv(64, n_classes)

    def forward(self, x):
        x1, out = self.down1(x)
        x2, out = self.down2(out)
        x3, out = self.down3(out)
        x4, out = self.down4(out)
        out = self.bottleneck(out)
        out = self.up1(out, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.classifier(out)
        return out 
    