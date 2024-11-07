"""Taken from https://www.kaggle.com/code/drvaibhavkumar/alexnet-in-pytorch-cifar10-clas-83-test-accuracy"""

import torch
import torch.nn as nn


def getModel(pretrained_path: str = ""):
    if pretrained_path != "":
        model = torch.load(pretrained_path)
        return model
    AlexNet_Model = torch.hub.load("pytorch/vision:v0.6.0", "alexnet", pretrained=True)
    AlexNet_Model.classifier[1] = nn.Linear(9216, 4096)
    AlexNet_Model.classifier[4] = nn.Linear(4096, 1024)
    AlexNet_Model.classifier[6] = nn.Linear(1024, 10)

    return AlexNet_Model
