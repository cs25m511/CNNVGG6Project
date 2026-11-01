# Models/NeuralNet.py

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, activation="relu", num_classes=10, use_batchnorm=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        activations = {
            "relu": nn.ReLU(inplace=True),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),
            "gelu": nn.GELU()
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        self.act = activations[activation]

        # Conv Layers (bias=False when using BatchNorm)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=not use_batchnorm)

        # Conditional BatchNorm Layers
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn_fc1 = nn.BatchNorm1d(256)
            self.bn_fc2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Dense Layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256, bias=not use_batchnorm)
        self.fc2 = nn.Linear(256, 128, bias=not use_batchnorm)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.act(x)

        # Conv Block 2
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)

        # Conv Block 3
        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = self.act(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        # FC 1
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn_fc1(x)
        x = self.act(x)
        x = self.dropout1(x)

        # FC 2
        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.bn_fc2(x)
        x = self.act(x)
        x = self.dropout2(x)

        # Output Layer
        x = self.fc3(x)
        return x