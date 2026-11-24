import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self, channels=1, size=28, hidden=(512, 256), classes=10):
        super().__init__()
        layers = [nn.Flatten()]
        width = channels * size * size
        for value in hidden:
            layers += [nn.Linear(width, value), nn.ReLU(), nn.Dropout(0.2)]
            width = value
        layers.append(nn.Linear(width, classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
