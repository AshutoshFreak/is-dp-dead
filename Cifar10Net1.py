from BasicBlock import BasicBlock
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class ResNet1(nn.Module):

    def __init__(
        self,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(2, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # def _resnet1(
    #     layers: List[int],
    #     **kwargs: Any
    # ) -> ResNet1:
    #     model = ResNet1(layers, **kwargs)
    #     return model
    #
    #
    # def resnet18_1(
    #     pretrained: bool = False, progress: bool = True, **kwargs: Any
    # ) -> ResNet1:
    #     r"""ResNet-18 model from
    #     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    #     Args:
    #         pretrained (bool): If True, returns a model pre-trained on ImageNet
    #         progress (bool): If True, displays a progress bar of the download to stderr
    #     """
    #     return _resnet1(
    #         [2, 2, 2, 2], **kwargs
    #     )
    #


def getModel():
    return ResNet1()
