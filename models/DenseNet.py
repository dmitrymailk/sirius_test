import torch
from torch import nn
    
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(self.conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def conv_block(self, input_channels, num_channels):
        return nn.Sequential(
            nn.Dropout(0.1),
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X

class DenseNet(nn.Module):
    def __init__(self, n_classes=10):
        super(DenseNet, self).__init__()

        num_channels, growth_rate = 128, 32

        b1 = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_convs_in_dense_blocks = [8, 8, 8, 8]
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # This is the number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added between
            # the dense blocks
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.conv = nn.Sequential(
            b1, 
            *blks,
            nn.BatchNorm2d(num_channels), 
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.BatchNorm1d(num_channels),
            nn.Linear(num_channels, n_classes)
        )
    
    def transition_block(self, input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


