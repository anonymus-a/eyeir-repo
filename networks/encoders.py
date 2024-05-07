import math
import queue
from turtle import forward
import torch
import threading
import collections
from abc import ABC
from torch import nn
import torch.nn.functional as F
import torchvision.models as tm
from torch.autograd import Variable
from torch.nn.functional import pad, upsample
import torch.utils.model_zoo as tumz
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.autograd import grad as ta_grad
from torchvision.models.resnet import model_urls
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast



def get_encoder(cfg):
    if cfg.name == 'resnet18':
        return ResNet18EncoderMS(cfg.pretrained)
    else:
        raise NotImplementedError(cfg)

class ResNet18EncoderMS(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.backbone = tm.resnet18(pretrained=pretrained)

    def forward(self, x):
        x_in = x.clone()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x1 = self.backbone.maxpool(x)

        x2 = self.backbone.layer1(x1)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x5 = self.backbone.layer4(x4)

        out_feature = {
            'input': x_in,  
            'shallow': x1,  
            'low': x2,  
            'mid': x3,  
            'deep': x4,  
            'out': x5,  
            'name': 'resnet18'
        }

        return out_feature

