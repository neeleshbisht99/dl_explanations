import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Conv3DBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool3d(2)
        self.batchnorm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.batchnorm(x)
        return x

class CNN3D(nn.Module):
    def __init__(self, width=128, height=128, depth=64):
        super(CNN3D, self).__init__()
        self.conv1 = Conv3DBlock(1, 64)
        self.conv2 = Conv3DBlock(64, 64)
        self.conv3 = Conv3DBlock(64, 128)
        self.conv4 = Conv3DBlock(128, 256)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(128, 2) # number of classes: 2 


    def forward(self, x):
        features_blobs = []
        x = self.conv1(x)
        features_blobs.append(x)
        x = self.conv2(x)
        features_blobs.append(x)
        x = self.conv3(x)
        features_blobs.append(x)
        x = self.conv4(x)
        features_blobs.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, features_blobs

