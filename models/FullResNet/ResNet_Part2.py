import torch
import torch.nn as nn


class ResNet_Part2(nn.Module):
    def __init__(self):
        super(ResNet_Part2, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 9)

    def forward(self, x):

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
