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


class SSEModule(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.se = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class CSEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore
        x = x * self.se(x)
        return x


class SCSEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.c_se = CSEModule(in_channels, reduction)
        self.s_se = SSEModule(in_channels)

    def forward(self, x):  # type: ignore
        return self.c_se(x) + self.s_se(x)


class SENextBottleneck(nn.Module):
    pool: t.Union[None, nn.MaxPool2d, nn.AvgPool2d]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 32,
        reduction: int = 16,
        pool: t.Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__()
        mid_channels = groups * (out_channels // 2 // groups)
        self.conv1 = ConvBR2d(in_channels, mid_channels, 1, 0, 1,)
        self.conv2 = ConvBR2d(mid_channels, mid_channels, 3, 1, 1, groups=groups)
        self.conv3 = ConvBR2d(mid_channels, out_channels, 1, 0, 1, is_activation=False)
        self.se = CSEModule(out_channels, reduction)
        self.stride = stride
        if in_channels != out_channels:
            self.is_shortcut = True
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
        #
        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool2d(x, self.stride, self.stride)  # avg
            x = self.shortcut(x)
        x = x + s
        x = F.relu(x, inplace=True)

        return x


class ConvBR2d(nn.Module):
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
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # type: ignore
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # type: ignore
        return self.double_conv(x)


class SENeXt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        width: int,
        ratio: float = 2.0,
        stride: int = 4,
    ) -> None:
        super().__init__()
        self.in_conv = ConvBR2d(in_channels, width, is_activation=False)
        self.layer = nn.Sequential(
            OrderedDict(
                {
                    f"layer-{i}": SENextBottleneck(
                        in_channels=int(width * ratio ** i),
                        out_channels=int(width * ratio ** (i + 1)),
                        stride=stride,
                        groups=int(width * ratio ** (i + 1)) // depth,
                    )
                    for i in range(depth)
                }
            )
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(int(width * ratio ** depth), out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore
        x = self.in_conv(x)
        x = self.layer(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x
