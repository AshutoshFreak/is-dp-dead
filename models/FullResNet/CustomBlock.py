import torch.nn as nn


class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomBlock, self).__init__()
        # Define conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        # Define bn1 (GroupNorm with 2 groups for in_channels channels)
        # self.gn1 = nn.GroupNorm(2, in_channels, eps=1e-05, affine=True)
        # Define ReLU activation
        self.relu = nn.ReLU(inplace=True)
        # Define conv2
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        # Define bn2 (GroupNorm with 2 groups for out_channels channels)
        self.gn2 = nn.GroupNorm(2, out_channels, eps=1e-05, affine=True)
        # Define downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), bias=False
            ),
            nn.GroupNorm(32, out_channels, eps=1e-05, affine=True),
        )

    def forward(self, x):
        # Forward pass through conv1, bn1, and relu
        out = self.conv1(x)
        # out = self.gn1(out)
        out = self.relu(out)
        # Forward pass through conv2 and bn2
        out = self.conv2(out)
        out = self.gn2(out)

        # Downsampling path
        residual = self.downsample(x)

        # Add residual to the output
        out += residual

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)

        return out
