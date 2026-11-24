import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def forward(self, x):
        return self.conv(x) * self.scale


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MixedPool2d(nn.Module):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        w = torch.sigmoid(self.weight)
        return w * F.max_pool2d(x, self.kernel_size) + (1 - w) * F.avg_pool2d(x, self.kernel_size)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        middle = max(1, channels // 4)
        self.net = nn.Sequential(nn.Conv2d(channels, middle, 1), nn.ReLU(), nn.Conv2d(middle, middle, 3, padding=1), nn.ReLU(), nn.Conv2d(middle, channels, 1))

    def forward(self, x):
        return F.relu(x + self.net(x))


class WideBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, padding=1), nn.ReLU(), nn.Conv2d(channels * 2, channels, 3, padding=1))

    def forward(self, x):
        return F.relu(x + self.net(x))
