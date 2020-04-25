import numpy as np

import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 1, gamma: float = 2, reduce: bool = True,
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):  # type:ignore
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class SSE1d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.se = nn.Sequential(nn.Conv1d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class SSEModule(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.se = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class CSE1d(nn.Module):
    def __init__(self, in_channels: int, reduction: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class SCSE1d(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.c_se = CSE1d(in_channels, reduction)
        self.s_se = SSE1d(in_channels)

    def forward(self, x):  # type: ignore
        return self.c_se(x) + self.s_se(x)


class SENextBottleneck1d(nn.Module):
    pool: t.Union[None, nn.MaxPool1d, nn.AvgPool1d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 32,
        reduction: int = 16,
        pool: t.Literal["max", "avg"] = "max",
        is_shortcut: bool = False,
    ) -> None:
        super().__init__()
        mid_channels = groups * (out_channels // 2 // groups)
        self.conv1 = ConvBR1d(in_channels, mid_channels, 1, 0, 1,)
        self.conv2 = ConvBR1d(mid_channels, mid_channels, 3, 1, 1, groups=groups)
        self.conv3 = ConvBR1d(mid_channels, out_channels, 1, 0, 1, is_activation=False)
        self.se = CSE1d(out_channels, reduction)
        self.stride = stride
        self.is_shortcut = is_shortcut
        if self.is_shortcut:
            self.shortcut = ConvBR1d(
                in_channels, out_channels, 1, 0, 1, is_activation=False
            )
        self.activation = nn.ReLU(inplace=True)
        if stride > 1:
            if pool == "max":
                self.pool = nn.MaxPool1d(stride, stride)
            elif pool == "avg":
                self.pool = nn.AvgPool1d(stride, stride)

    def forward(self, x):  # type: ignore
        s = self.conv1(x)
        s = self.conv2(s)
        if self.stride > 1:
            s = self.pool(s)
        s = self.conv3(s)
        s = self.se(s)
        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool1d(x, self.stride, self.stride)  # avg
            x = self.shortcut(x)
        x = x + s
        x = self.activation(x)

        return x


class ConvBR1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
        is_activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.is_activation = is_activation

        if is_activation:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):  # type: ignore
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.activation(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels: int, out_channels: int, pool: t.Literal["max", "avg"] = "max"
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            SENextBottleneck1d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                is_shortcut=True,
                pool=pool,
            ),
            SENextBottleneck1d(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                is_shortcut=False,
            ),
        )

    def forward(self, x):  # type: ignore
        return self.block(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    up: t.Union[nn.Upsample, nn.ConvTranspose1d]

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = False,
    ) -> None:
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        self.conv1 = ConvBR1d(
            out_channels + out_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv2 = ConvBR1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):  # type: ignore
        x1 = self.up(x1)
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        channels = np.array([128, 256, 512, 1024, 2048])

        self.in_channels = in_channels
        self.inc = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            ConvBR1d(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            ConvBR1d(
                in_channels=channels[0],
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBR1d(
                in_channels=channels[0],
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.down1 = Down(channels[0], channels[1], pool="max")
        self.down2 = Down(channels[1], channels[2], pool="max")
        self.down3 = Down(channels[2], channels[3], pool="max")
        self.down4 = Down(channels[3], channels[4], pool="avg")
        self.up1 = Up(channels[-1], channels[-2])
        self.up2 = Up(channels[-2], channels[-3])
        self.up3 = Up(channels[-3], channels[-4])
        self.up4 = Up(channels[-4], channels[-5])
        self.outc = nn.Sequential(
            nn.Conv1d(channels[-5], out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):  # type: ignore
        x1 = self.inc(x)  # [B, 64, L]
        x2 = self.down1(x1)  # [B, 128, L//2]
        x3 = self.down2(x2)  # [B, 256, L//4]
        x4 = self.down3(x3)  # [B, 512, L//8]
        x5 = self.down4(x4)  # [B, 1024, L//8]
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
