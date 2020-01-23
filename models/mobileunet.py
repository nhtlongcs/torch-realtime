import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision.models import mobilenet_v2


__all__ = ['MobileUnet']


class MobileUnet(nn.Module):
    """Some Information about MobileUnet"""

    def __init__(self, input_size=(224,224,3)):
        super(Unet, self).__init__()

        self.mobilenet = mobilenet_v2(pretrained=True, progress=True)
        for param in self.mobilenet.parameters():
            param.requires_grad_(False)

        # sizes = [320] + [160]*3 + [96]*3 + [64]*4 + [32]*3 + [24]*2 + [16]
        self.up1 = Up(self.mobilenet.classifier.in_channels, 96, 96)
        self.up2 = Up(96, 32, 32)
        self.up3 = Up(32, 24, 24)
        self.up4 = Up(24, 16, 16)

        self.conv_last = nn.Conv2d(16, 3, 1)
        self.conv_score = nn.Conv2d(3, 1, 1)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, Up):
                module._init_weights()
            

    def forward(self, x):
        for n in range(0, 2):
            x = self.mobilenet.features[n](x)
        x1 = x
        logging.debug((x1.shape, 'x1')) # [B, 16, H, W]

        for n in range(2, 4):
            x = self.mobilenet.features[n](x)
        x2 = x
        logging.debug((x2.shape, 'x2')) # [B, 24, H, W]

        for n in range(4, 7):
            x = self.mobilenet.features[n](x)
        x3 = x
        logging.debug((x3.shape, 'x3')) # [B, 32, H, W]

        for n in range(7, 14):
            x = self.mobilenet.features[n](x)
        x4 = x
        logging.debug((x4.shape, 'x4')) # [B, 96, H, W]

        for n in range(14, 19):
            x = self.mobilenet.features[n](x)
        x5 = x
        logging.debug((x5.shape, 'x5')) # [B, 1280, H, W]

        up1 = torch.cat([x4,self.dconv1(x)], dim=1)
        up1 = self.invres1(up1)
        logging.debug((up1.shape, 'up1'))

        self.up2.forward(x, )
        logging.debug((up2.shape, 'up2'))

        up3 = torch.cat([
            x2,
            self.dconv3(up2)
        ], dim=1)
        up3 = self.invres3(up3)
        logging.debug((up3.shape, 'up3'))

        up4 = torch.cat([
            x1,
            self.dconv4(up3)
        ], dim=1)
        up4 = self.invres4(up4)
        logging.debug((up4.shape, 'up4'))

        x = self.conv_last(up4)
        logging.debug((x.shape, 'conv_last'))

        x = self.conv_score(x)
        logging.debug((x.shape, 'conv_score'))

        # x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # logging.debug((x.shape, 'interpolate'))

        x = torch.sigmoid(x)


        return x

class Up(nn.Module):
    def __init__(self, in_channels, in_concat_channels, out_channels):
        self.in_channels = in_channels
        self.in_concat_channels = in_concat_channels
        self.out_channels = out_channels

        self.up_conv = UpConv(in_channels)
        self.double_conv = DoubleConv(self.in_channels + self.in_concat_channels, self.out_channels)

    def forward(self, x1, x2):
        '''
        :param x1: feature from previous layer
        :param x2: feature to concat
        '''
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_conv = nn.Sequential(OrderedDict([
            ('up', nn.Upsample(scale_factor=2)),
            ('conv', nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)),
        ]))
    
    def forward(self, x):
        return self.up_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
