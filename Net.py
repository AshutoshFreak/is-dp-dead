import Cifar10Net1
import Cifar10Net2
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Cifar10Net1.getModel()

    def forward(self, x):
        x = self.model(x)
        return x


class Decider(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Cifar10Net2.getModel()

    def forward(self, x):
        x = self.model(x)
        return x


class Net(nn.Module):
    def __init__(self, encoder, decider):
        super().__init__()
        self.encoder = encoder
        self.decider = decider

    def forward(self, x):
        x = self.encoder(x)
        x = self.decider(x)
        return x


def getModel():
    encoder = Encoder()
    decider = Decider()
    return Net(encoder, decider)
