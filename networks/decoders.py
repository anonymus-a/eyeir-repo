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

from .blocks import Conv2dBlock, ResidualBlock, PAM_Module, PyramidPooling

def get_decoder(input_dim, cfg, type='FC'):
    if type == 'FC':
        return FCDecoder(input_dim, cfg.output_dim, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                         res_scale=cfg.res_scale, n_resblocks=cfg.n_resblocks,
                         se_reduction=cfg.se_reduction == 1, pyramid=cfg.pyramid == 1, end_act=cfg.end_act)
    elif type == 'FC_v2':
        return FCDecoder_v2(input_dim, cfg.output_dim, cfg.fc_input, intermediate_dim=cfg.intermediate_dim,
                            norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                            res_scale=cfg.res_scale, n_resblocks=cfg.n_resblocks,
                            se_reduction=cfg.se_reduction == 1, pyramid=cfg.pyramid == 1, end_act=cfg.end_act)
    elif type == 'FC_sep':
        return [
            FCDecoder_ab(input_dim, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                         res_scale=cfg.res_scale, n_resblocks=cfg.n_resblocks,
                         se_reduction=cfg.se_reduction == 1),
            FCDecoder_c(cfg.output_dim, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                         pyramid=cfg.pyramid == 1, end_act=cfg.end_act)
        ]

    elif type == 'FC_v2_sep':
        return [
            FCDecoder_ab(input_dim, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                         res_scale=cfg.res_scale, n_resblocks=cfg.n_resblocks,
                         se_reduction=cfg.se_reduction == 1),
            FCDecoder_v2_c(cfg.output_dim, cfg.fc_input, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                         pyramid=cfg.pyramid == 1, end_act=cfg.end_act)
        ]
    elif type == 'FC_v2_sep_attention':
        return [
            FCDecoder_ab(input_dim, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                         res_scale=cfg.res_scale, n_resblocks=cfg.n_resblocks,
                         se_reduction=cfg.se_reduction == 1),
            FCDecoder_v2_c(cfg.output_dim, cfg.fc_input, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                         pyramid=cfg.pyramid == 1, end_act=cfg.end_act, use_self_attention=True)
        ]
    elif type == 'FC_attention_heavy':
        return FCDecoder_attention(input_dim, cfg.output_dim, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type,
                         res_scale=cfg.res_scale, n_resblocks=cfg.n_resblocks,
                         se_reduction=cfg.se_reduction == 1, pyramid=cfg.pyramid == 1, end_act=cfg.end_act, use_self_attention=True)
    elif type == 'FullConnected': 
        return FullConnectedDecoder(input_dim, cfg.output_dim, intermediate_dim=cfg.intermediate_dim,
                         norm=cfg.norm, activation=cfg.activation, pad_type=cfg.pad_type)

    else:
        raise NotImplementedError

class FCDecoder_ab(nn.Module):

    def __init__(self, input_dim, intermediate_dim=64, norm="in", activation="lrelu",
                 pad_type="reflect", n_resblocks=7,
                 se_reduction=None, res_scale=1):
        super(FCDecoder_ab, self).__init__()
        model = []
        model += []
        model += [
        ]

        self.fc_prefix = nn.Sequential(
            Conv2dBlock(input_dim, intermediate_dim, [1, 1], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1),
            Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1),
            Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 2, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1)  # 224x224->112x112
        )

        dilation_config = [1] * n_resblocks

        self.fc_mediate = nn.ModuleList([ResidualBlock(
            intermediate_dim, dilation=dilation_config[i], norm=norm, act=activation,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)
        ])

    def forward(self, x, with_feat=False):
        feats = []
        x = self.fc_prefix(x)

        for block in self.fc_mediate:
            x = block(x)
            feats.append(x)

        if with_feat:
            return x, feats
        else:
            return x

class FCDecoder_c(nn.Module):


    def __init__(self, output_dim, intermediate_dim=64, norm="in", activation="lrelu", end_act='none',
                 pad_type="reflect", pyramid=False, use_self_attention=False):
        super(FCDecoder_c, self).__init__()
        self.use_sa = use_self_attention
        if self.use_sa:
            self.sa = PAM_Module(intermediate_dim)
        if pyramid:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                PyramidPooling(intermediate_dim, intermediate_dim, scales=(4, 8, 16, 32),
                               ct_channels=intermediate_dim // 4),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )
        else:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )

    def forward(self, x):
        if self.use_sa:
            x = self.sa(x)
        x = self.fc_postfix(x)
        return x

class FCDecoder_v2_c(nn.Module):

    def __init__(self, output_dim, fc_input, intermediate_dim=64, norm="in", activation="lrelu",
                 end_act='none', pad_type="reflect", pyramid=False, use_self_attention=False):
        super(FCDecoder_v2_c, self).__init__()
        self.use_sa = use_self_attention
        if self.use_sa:
            self.sa = PAM_Module(intermediate_dim)
        if pyramid:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                PyramidPooling(intermediate_dim, intermediate_dim, scales=(4, 8, 16, 32),
                               ct_channels=intermediate_dim // 4),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )
        else:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_basis = nn.Sequential(
            nn.Linear(fc_input, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.use_sa:
            x = self.sa(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        basis = self.fc_basis(y)

        x_dim1 = self.fc_postfix(x)

        basis = basis.view(b, 3, 1, 1)

        x_dim3 = x_dim1 * basis

        return x_dim3


class FullConnectedDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, intermediate_dim=64, norm="in", activation="lrelu",
                 pad_type="reflect"):
        super(FullConnectedDecoder, self).__init__()

        self.intermediate_dim = intermediate_dim
        self.conv = Conv2dBlock(input_dim, intermediate_dim, [1, 1], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(intermediate_dim, output_dim)

    def forward(self, x, with_feat=False):
        out = self.conv(x)
        out = self.pool(out)
        out = out.view(-1, self.intermediate_dim)
        out = self.fc(out)
        return out

class FCDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, intermediate_dim=64, norm="in", activation="lrelu", end_act='none',
                 pad_type="reflect", n_resblocks=7,
                 se_reduction=None, res_scale=1, pyramid=False, use_self_attention=False):
        super(FCDecoder, self).__init__()
        self.use_sa = use_self_attention
        model = []
        model += []
        model += [
        ]

        self.fc_prefix = nn.Sequential(
            Conv2dBlock(input_dim, intermediate_dim, [1, 1], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1),
            Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1),
            Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 2, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1)
        )
        model = []
        dilation_config = [1] * n_resblocks

        self.fc_mediate = nn.ModuleList([ResidualBlock(
            intermediate_dim, dilation=dilation_config[i], norm=norm, act=activation,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)
        ])
        
        if self.use_sa:
            self.sa = PAM_Module(intermediate_dim)

        if pyramid:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                PyramidPooling(intermediate_dim, intermediate_dim, scales=(4, 8, 16, 32),
                               ct_channels=intermediate_dim // 4),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )
        else:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )

    def forward(self, x, with_feat=False):
        feats = []
        x = self.fc_prefix(x)

        for block in self.fc_mediate:
            x = block(x)
            feats.append(x)
        
        if self.use_sa:
            x = self.sa(x)

        x = self.fc_postfix(x)
        if with_feat:
            return x, feats
        else:
            return x

class FCDecoder_v2(nn.Module):

    def __init__(self, input_dim, output_dim, fc_input, intermediate_dim=64, norm="in", activation="lrelu",
                 end_act='none', pad_type="reflect", n_resblocks=7,
                 se_reduction=None, res_scale=1, pyramid=False, use_self_attention=False):
        super(FCDecoder_v2, self).__init__()
        self.use_sa = use_self_attention
        model = []
        model += []
        model += [
        ]

        self.fc_prefix = nn.Sequential(
            Conv2dBlock(input_dim, intermediate_dim, [1, 1], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1),
            Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1),
            Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 2, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1) 
        )
        model = []
        dilation_config = [1] * n_resblocks

        self.fc_mediate = nn.ModuleList([ResidualBlock(
            intermediate_dim, dilation=dilation_config[i], norm=norm, act=activation,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)
        ])
        if self.use_sa:
            self.sa = PAM_Module(intermediate_dim)

        if pyramid:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                PyramidPooling(intermediate_dim, intermediate_dim, scales=(4, 8, 16, 32),
                               ct_channels=intermediate_dim // 4),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )
        else:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_basis = nn.Sequential(
            nn.Linear(fc_input, 3),
            nn.Sigmoid()
        )

    def forward(self, x, with_feat=False):
        feats = []
        x = self.fc_prefix(x)

        for block in self.fc_mediate:
            x = block(x)
            feats.append(x)
        
        if self.use_sa:
            x = self.sa(x)

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        basis = self.fc_basis(y)

        x_dim1 = self.fc_postfix(x)

        basis = basis.view(b, 3, 1, 1)

        x_dim3 = x_dim1 * basis

        if with_feat:
            return x_dim3, feats
        else:
            return x_dim3

class FCDecoder_attention(nn.Module):

    def __init__(self, input_dim, output_dim, intermediate_dim=64, norm="in", activation="lrelu", end_act='none',
                 pad_type="reflect", n_resblocks=7,
                 se_reduction=None, res_scale=1, pyramid=False, use_self_attention=False):
        super(FCDecoder_attention, self).__init__()
        self.use_sa = use_self_attention
        model = []
        model += []
        model += [
        ]

        self.fc_prefix = nn.Sequential(
            Conv2dBlock(input_dim, intermediate_dim, [1, 1], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1),
            Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1),
            Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 2, 'SAME', norm=norm, activation=activation,
                        pad_type=pad_type,
                        dilation=1) 
        )
        model = []
        dilation_config = [1] * n_resblocks

        self.fc_mediate = nn.ModuleList([ResidualBlock(
            intermediate_dim, dilation=dilation_config[i], norm=norm, act=activation,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)
        ])
        
        self.sa = PAM_Module(intermediate_dim)

        if pyramid:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                PyramidPooling(intermediate_dim, intermediate_dim, scales=(4, 8, 16, 32),
                               ct_channels=intermediate_dim // 4),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )
        else:
            self.fc_postfix = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2dBlock(intermediate_dim, intermediate_dim, [3, 3], 1, 'SAME', norm=norm, activation=activation,
                            pad_type=pad_type,
                            dilation=1),
                Conv2dBlock(intermediate_dim, output_dim, [1, 1], 1, 0, norm=norm, activation=end_act,
                            pad_type=pad_type,
                            dilation=1)
            )

    def forward(self, x, with_feat=False):
        feats = []
        x = self.fc_prefix(x)

        for block in self.fc_mediate:
            x = block(x)
            x = self.sa(x)
            feats.append(x)
            
        x = self.fc_postfix(x)
        if with_feat:
            return x, feats
        else:
            return x