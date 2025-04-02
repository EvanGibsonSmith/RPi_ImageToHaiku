import torch
import torch.nn as nn

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 7, 11], dilations=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=k, stride=1, dilation=d)
            for k, d in zip(kernel_sizes, dilations)
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=k, stride=1)
            for k in kernel_sizes
        ])

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x = x + conv2(conv1(x))
        return x

class HifiganGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv1d(80, 512, kernel_size=7, stride=1)
        
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        ])

        self.resblocks = nn.ModuleList([
            ResBlock1(256), ResBlock1(256, [7, 7, 7]), ResBlock1(256, [11, 11, 11]),
            ResBlock1(128), ResBlock1(128, [7, 7, 7]), ResBlock1(128, [11, 11, 11]),
            ResBlock1(64), ResBlock1(64, [7, 7, 7]), ResBlock1(64, [11, 11, 11]),
            ResBlock1(32), ResBlock1(32, [7, 7, 7]), ResBlock1(32, [11, 11, 11])
        ])

        self.conv_post = nn.Conv1d(32, 1, kernel_size=7, stride=1)

    def forward(self, x):
        x = self.conv_pre(x)
        for up, res in zip(self.ups, self.resblocks):
            x = up(x)
            x = res(x)
        x = self.conv_post(x)
        return x
