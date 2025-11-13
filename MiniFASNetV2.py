import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(MiniFASNetV2, self).__init__()
        # Convolutional layers
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(16)
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn_2 = nn.BatchNorm2d(32)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn_5 = nn.BatchNorm2d(256)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))

        # Global average pooling
        x = self.global_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc(x)

        # ✅ return output properly
        return x
