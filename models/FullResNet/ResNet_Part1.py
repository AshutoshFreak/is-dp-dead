import torch.nn as nn


class ResNet_Part1(nn.Module):
    def __init__(self):
        super(ResNet_Part1, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.GroupNorm(2, 64, eps=1e-5, affine=True),
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.GroupNorm(2, 64, eps=1e-5, affine=True),
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.GroupNorm(2, 64, eps=1e-5, affine=True),
            nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.GroupNorm(2, 64, eps=1e-5, affine=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        return x
