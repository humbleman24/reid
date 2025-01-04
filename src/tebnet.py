import torch
import torch.nn as nn
import torchvision.models as models

import os


class TEBNet(nn.modules):
    def __init__(self):
        super(TEBNet, self).__init__()
        self.backbone = self._get_backbone()

        self.global_branch = self._build_global_branch()
    
    def _get_backbone(self):
        resnet = models.resnet50(pretrained=True)
        layer4 = resnet.layer4[0]
        # change the stride of the first conv layer in layer4
        # only place the stride is applied in the model
        layer4.conv1.stride = (1, 1)
        return nn.Sequential(*list(resnet.children())[:-2])

    def _build_global_branch(self):
        global_sequence = nn.Sequential([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, kernel_size = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])

        return global_sequence
    
    def _build_complementary_branch(self):
        sim_unit = nn.AdaptiveAvgPool2d((4, 1))

        





