import numpy as np

import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import LeakyReLU as Activation


class ConvBR2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        stride: int = 1,
        is_activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.is_activation = is_activation

        if is_activation:
            self.activation = Activation(inplace=True)

    def forward(self, x):  # type: ignore
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.activation(x)
        return x


class SSE2d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.se = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class CSE2d(nn.Module):
    def __init__(self, in_channels: int, reduction: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            Activation(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class SENextBottleneck2d(nn.Module):
    pool: t.Union[None, nn.MaxPool2d, nn.AvgPool2d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 8,
        groups: int = 16,
        pool: t.Literal["max", "avg"] = "max",
        is_shortcut: bool = True,
    ) -> None:
        super().__init__()
        mid_channels = groups * (out_channels // 2 // groups)
        self.conv1 = ConvBR2d(in_channels, mid_channels, 1, 0, 1,)
        self.conv2 = ConvBR2d(mid_channels, mid_channels, 3, 1, 1, groups=groups)
        self.conv3 = ConvBR2d(mid_channels, out_channels, 1, 0, 1, is_activation=False)
        self.se = CSE2d(out_channels, reduction)
        self.stride = stride
        self.is_shortcut = is_shortcut
        self.activation = Activation(inplace=True)
        if self.is_shortcut:
            self.shortcut = ConvBR2d(
                in_channels, out_channels, 1, 0, 1, is_activation=False
            )
        if stride > 1:
            if pool == "max":
                self.pool = nn.MaxPool2d(stride, stride)
            elif pool == "avg":
                self.pool = nn.AvgPool2d(stride, stride)

    def forward(self, x):  # type: ignore
        s = self.conv1(x)
        s = self.conv2(s)
        if self.stride > 1:
            s = self.pool(s)
        s = self.conv3(s)
        s = self.se(s)
        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool2d(x, self.stride, self.stride)  # avg
            x = self.shortcut(x)
        x = x + s
        x = self.activation(x)
        return x


class Down2d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels: int, out_channels: int, pool: t.Literal["max", "avg"] = "max"
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            SENextBottleneck2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                is_shortcut=True,
                pool=pool,
            ),
            SENextBottleneck2d(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                is_shortcut=False,
            ),
        )

    def forward(self, x):  # type: ignore
        return self.block(x)


class Up2d(nn.Module):
    """Upscaling then double conv"""

    up: t.Union[nn.Upsample, nn.ConvTranspose2d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
        merge: bool = True,
    ) -> None:
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.merge = merge
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
        if self.merge:
            self.conv1 = ConvBR2d(
                in_channels + out_channels, in_channels, kernel_size=3, padding=1
            )
        else:
            self.conv1 = ConvBR2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = SENextBottleneck2d(in_channels, out_channels)

    def forward(self, x1, x2):  # type: ignore
        x1 = self.up(x1)
        diff_h = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff_w = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, (diff_h - diff_h // 2, diff_w - diff_w // 2))
        if self.merge:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base_channel = 32

        self.inc = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=base_channel,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
        )
        self.down1 = Down2d(base_channel, base_channel * 2, pool="max")
        self.down2 = Down2d(base_channel * 2, base_channel * 4, pool="max")
        self.up2 = Up2d(base_channel * 4, base_channel * 2, bilinear=False, merge=True)
        self.up1 = Up2d(base_channel * 2, base_channel, bilinear=False, merge=True)

        self.outc = nn.Sequential(
            SENextBottleneck2d(base_channel, base_channel),
            SENextBottleneck2d(base_channel, base_channel),
            nn.Conv2d(base_channel, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):  # type: ignore
        input_shape = x.shape
        x = x.view(x.shape[0], 1, *x.shape[1:])
        x = torch.log(x)
        n0 = self.inc(x)  # [B, 64, L]
        n1 = self.down1(n0)  # [B, 128, L//2]
        n2 = self.down2(n1)  # [B, 128, L//2]
        n = self.up2(n2, n1)
        n = self.up1(n1, n0)
        n = self.outc(n)
        x = torch.exp(n + x)
        x = x.view(*input_shape)
        return x
