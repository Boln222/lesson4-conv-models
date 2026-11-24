import torch.nn as nn
from .custom_layers import ChannelAttention


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.Dropout2d(dropout), nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class CNN(nn.Module):
    def __init__(self, channels=1, classes=10, depth=2, kernel=3, regularized=False):
        super().__init__()
        blocks, current = [], channels
        for i in range(depth):
            out = 32 * (2 ** min(i, 1))
            blocks += [nn.Conv2d(current, out, kernel, padding=kernel // 2), nn.BatchNorm2d(out), nn.ReLU()]
            if regularized:
                blocks.append(nn.Dropout2d(0.2))
            if i % 2:
                blocks.append(nn.MaxPool2d(2))
            current = out
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(current, classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class ResidualCNN(nn.Module):
    def __init__(self, channels=1, classes=10, blocks=2, attention=False, regularized=False):
        super().__init__()
        drop = 0.2 if regularized else 0
        body = [nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        body += [ResidualBlock(32, drop) for _ in range(blocks)]
        if attention:
            body.append(ChannelAttention(32))
        body += [nn.AdaptiveAvgPool2d(1)]
        self.features = nn.Sequential(*body)
        self.head = nn.Linear(32, classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def kernel_model(kernel, channels=1, classes=10):
    if kernel == 'mixed':
        return nn.Sequential(nn.Conv2d(channels, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, classes))
    return CNN(channels, classes, depth=2, kernel=int(kernel))
