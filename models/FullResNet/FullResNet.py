import torch.nn as nn
from models.FullResNet.ResNet_Part1 import ResNet_Part1
from models.FullResNet.ResNet_Part2 import ResNet_Part2
from models.FullResNet.CustomBlock import CustomBlock


class FullResNet(nn.Module):
    def __init__(
        self, model_part1, model_layer2, model_layer3, model_layer4, model_part2
    ):
        super(FullResNet, self).__init__()
        # Store each part as a submodule
        self.model_part1 = model_part1
        self.model_layer2 = model_layer2
        self.model_layer3 = model_layer3
        self.model_layer4 = model_layer4
        self.model_part2 = model_part2

    def forward(self, x):
        # Pass the input through each part sequentially
        x = self.model_part1(x)  # Initial ResNet layers
        x = self.model_layer2(x)  # Custom block layer 2
        x = self.model_layer3(x)  # Custom block layer 3
        x = self.model_layer4(x)  # Custom block layer 4
        x = self.model_part2(x)  # Final ResNet layers
        return x


def getModel():
    model_part1 = ResNet_Part1()
    model_layer2 = CustomBlock(64, 128)
    model_layer3 = CustomBlock(128, 256)
    model_layer4 = CustomBlock(256, 512)
    model_part2 = ResNet_Part2()

    model = FullResNet(
        model_part1, model_layer2, model_layer3, model_layer4, model_part2
    )
    return model
