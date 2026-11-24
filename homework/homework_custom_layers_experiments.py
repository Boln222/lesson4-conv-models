import torch
import torch.nn as nn
from models import ScaledConv2d, ChannelAttention, Swish, MixedPool2d, ResidualBlock, BottleneckBlock, WideBlock
from utils import count_parameters


def check(name, layer, shape=(4, 16, 16, 16)):
    x = torch.randn(*shape, requires_grad=True)
    y = layer(x)
    y.mean().backward()
    print(f'{name}: input={tuple(x.shape)}, output={tuple(y.shape)}, parameters={count_parameters(layer)}, gradient={x.grad.norm().item():.6f}')


def main():
    check('ScaledConv2d', ScaledConv2d(16, 16))
    check('ChannelAttention', ChannelAttention(16))
    check('Swish', Swish())
    check('MixedPool2d', MixedPool2d())
    check('ResidualBlock', ResidualBlock(16))
    check('BottleneckBlock', BottleneckBlock(16))
    check('WideBlock', WideBlock(16))
    standard = nn.Conv2d(16, 16, 3, padding=1)
    custom = ScaledConv2d(16, 16)
    print(f'Conv2d={count_parameters(standard)}, ScaledConv2d={count_parameters(custom)}')


if __name__ == '__main__':
    main()
