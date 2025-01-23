import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # 1st Block
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5, padding=2
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)

        # 2nd Block
        self.conv2_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.dropout2 = nn.Dropout(p=0.2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 9 * 9, 512)  # Updated input size to 4096
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 1st Block
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # 2nd Block
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
