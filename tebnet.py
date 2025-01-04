import torch
import torch.nn as nn
from torchvision import models


def modify_resnet_stride(model):
    layer4 = model.layer4[0]
    layer4.conv1.stride = (1, 1)
    if layer4.downsample is not None:
        layer4.downsample[0].stride = (1, 1)

class TBE_Net(nn.Module):
    def __init__(self):
        super(TBE_Net, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modify_resnet_stride(resnet)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])     # 移除最后两个全连接层

        self.global_branch = self._build_global_branch()


    def _build_global_branch(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.linear(256, 10)
            )












