import torch
import torch.nn as nn
import math


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpSLayer(nn.Module):
    def __init__(self, dim, upscale) -> None:
        super().__init__()
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(upscale, dim)
    
    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        return x


class DownSLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ds_layer = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        x = self.ds_layer(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim=96, kernel_size=3, act_layer=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)

    def forward(self, x):
        _x = x
        x = self.act_layer(self.conv1(x))
        x = self.conv2(x) + _x
        return x


class MultiConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.convb1 = ConvBlock(dim=dim, kernel_size=3)
        self.convb2 = ConvBlock(dim=dim, kernel_size=5)
        self.convb3 = ConvBlock(dim=dim, kernel_size=7)
        self.pw_conv = nn.Conv2d(dim * 3, dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.convb1(x)
        x2 = self.convb2(x)
        x3 = self.convb3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.pw_conv(x)
        return x


class ReversedISP(nn.Module):
    def __init__(self, input_dim=3, embed_dim=96, num_out_ch=4):
        super().__init__()
        # shallow feature extractor
        self.shallow_conv = nn.Conv2d(input_dim, embed_dim, 3, 1, 1)

        # deep feature extractor
        self.dslayer1 = DownSLayer(dim=embed_dim) # 256
        self.multiconv1 = MultiConv(dim=embed_dim)
        self.dslayer2 = DownSLayer(dim=embed_dim) # 128
        self.multiconv2 = MultiConv(dim=embed_dim)
        self.dslayer3 = DownSLayer(dim=embed_dim) # 64
        self.multiconv3 = MultiConv(dim=embed_dim)

        self.uslayer1 = UpSLayer(dim=embed_dim, upscale=2) # 128
        self.multiconv4 = MultiConv(dim=embed_dim)
        self.uslayer2 = UpSLayer(dim=embed_dim, upscale=2) # 256

        # reconstrcution module
        self.last_conv = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.shallow_conv(x)

        x = self.dslayer1(x)
        _x1 = x
        
        x = self.multiconv1(x)
        x = self.dslayer2(x)
        _x2 = x

        x = self.multiconv2(x)
        x = self.dslayer3(x)
        x = self.multiconv3(x)

        x = self.uslayer1(x) + _x2
        x = self.multiconv4(x)
        x = self.uslayer2(x) + _x1

        x = self.last_conv(x)
        return x