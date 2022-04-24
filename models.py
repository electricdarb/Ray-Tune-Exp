import torch 
from torch import nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    """Simple Cnn for binary classification"""

    def __init__(self, image_size: int = 227, in_channels: int = 3, transforms = None, dropout_rate = .2) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size = 3, padding = 'same')
        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, padding = 'same')
        self.pool1 = nn.MaxPool2d(kernel_size = 2) # down samples to imagesize // 2 :(113)

        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, padding = 'same')
        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3, padding = 'same')
        self.pool2 = nn.MaxPool2d(kernel_size = 2) # down samples to imagesize // 2 :(56x56x64)

        self.dropout = nn.Dropout(dropout_rate)

        dense_dims = (image_size // 4) ** 2 * 32
        # max pool is applied with kernel of 2 3 times, so 
        self.dense = nn.Linear(dense_dims, 512)

        self.out = nn.Linear(512, 1)

    def forward(self, input: torch.tensor) -> torch.tensor:
        """forward pass through device"""
        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)

        # flatten while preserving batch 
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.dropout(x)

        # pass through a dense layer, mapping a binary from 0 to 1
        x = self.dense(x)
        x = F.relu(x)

        x = self.out(x)
        x = torch.sigmoid(x)

        # flatten x for loss func
        x = torch.flatten(x)

        return x





