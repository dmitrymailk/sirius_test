import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.drop_1 = nn.Dropout2d(0.1)

        self.prelu_1 = nn.ReLU()
        self.prelu_2 = nn.ReLU()

    def forward(self, X):
        Y = self.conv1(X)
        # Y = self.drop_1(Y)
        Y = self.bn1(Y)
        Y = self.prelu_1(Y)
        # Y = self.drop_1(Y)
        Y = self.conv2(Y)
        Y = self.bn2(Y)
        if self.conv3:
            X = self.conv3(X)
            # X = self.drop_1(X)
        Y += X
        return self.prelu_2(Y)

class ResNet(nn.Module):
    def __init__(self, n_classes=None):
        super(ResNet, self).__init__()

        b1 = nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(128), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b2 = nn.Sequential(*self.resnet_block(128, 128, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(128, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))

        self.conv = nn.Sequential(
            b1,
            # nn.Dropout2d(0.1), 
            b2, 
            # nn.Dropout2d(0.1), 
            b3, 
            # nn.Dropout2d(0.1), 
            b4, 
            # nn.Dropout2d(0.1), 
            b5
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(0.1),
            nn.Flatten(), 
            nn.BatchNorm1d(512),
            nn.Linear(512, n_classes)
        )

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x