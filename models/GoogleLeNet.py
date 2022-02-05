import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        
        # Path 2 
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
       
        # Path 3 
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        
        # Path 4 
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

        # batchnorms
        self.b_norm_1 = nn.BatchNorm2d(c1)
        
        self.b_norm_2_1 = nn.BatchNorm2d(c2[0])
        self.b_norm_2_2 = nn.BatchNorm2d(c2[1])

        self.b_norm_3_1 = nn.BatchNorm2d(c3[0])
        self.b_norm_3_2 = nn.BatchNorm2d(c3[1])

        self.b_norm_4 = nn.BatchNorm2d(c4)
        
        # regularization
        self.drop_1 = nn.Dropout2d(0.4)

    def forward(self, x):
        # path 1
        p1 = self.p1_1(x)
        p1 = self.drop_1(p1)
        p1 = self.b_norm_1(p1)
        p1 = F.relu(p1)

        # path 2
        p2 = self.p2_1(x)
        p2 = self.drop_1(p2)
        p2 = self.b_norm_2_1(p2)
        p2 = F.relu(p2)

        p2 = self.p2_2(p2)
        p2 = self.drop_1(p2)
        p2 = self.b_norm_2_2(p2)
        p2 = F.relu(p2)

        # path 3 
        p3 = self.p3_1(x)
        p3 = self.drop_1(p3)
        p3 = self.b_norm_3_1(p3)
        p3 = F.relu(p3)
        
        p3 = self.p3_2(p3)
        p3 = self.drop_1(p3)
        p3 = self.b_norm_3_2(p3)
        p3 = F.relu(p3)

        # path 4
        p4 = self.p4_1(x)
        p4 = self.p4_2(p4)
        p4 = self.drop_1(p4)
        p4 = self.b_norm_4(p4)
        p4 = F.relu(p4)
        
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)

class GoogleLeNet(nn.Module):
    def __init__(self, n_classes=None):
        super(GoogleLeNet, self).__init__()

        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # block 3
        block_3_inception_1 = 192, 64, (96, 128), (16, 32), 32
        block_3_inception_2 = 256, 128, (128, 192), (32, 96), 64

        # block 4
        block_4_inception_1 = 480, 192, (96, 208), (16, 48), 64

        # block 5
        block_5_inception_1 = 512, 256, (160, 320), (32, 128), 128
        
        self.block3 = nn.Sequential(
            Inception(*block_3_inception_1),
            Inception(*block_3_inception_2),
            nn.BatchNorm2d(480),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.block4 = nn.Sequential(
            Inception(*block_4_inception_1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.block5 = nn.Sequential(
            Inception(*block_5_inception_1),
            nn.BatchNorm2d(832),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())

        self.conv = nn.Sequential(self.block1, self.block2, self.block3, self.block4, self.block5)
        self.classifier = nn.Linear(832, self.n_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x





