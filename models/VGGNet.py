import torch
from torch import nn

class VGGNet(nn.Module):
    def __init__(self, n_classes=None, conv_arch=None):
        super(VGGNet, self).__init__()
        self.conv_arch = conv_arch
        self.n_classes = n_classes

        self.conv = self.get_conv()

        self.classifier = nn.Sequential(nn.Flatten(),
            # The fully-connected part
            nn.Linear(conv_arch[-1][1] * 4 * 4, 200), nn.ReLU(), nn.Dropout(0.4),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100), nn.ReLU(), nn.Dropout(0.4),
            nn.BatchNorm1d(100),
            nn.Linear(100, 10))
        
    def get_conv(self):
        conv_blks = []
        in_channels = 3
        # The convolutional part
        for (num_convs, out_channels) in self.conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(*conv_blks)

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Dropout(0.1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels
        layers.append(nn.AvgPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x