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

from .blocks import FusionBlock
from .encoders import get_encoder
from .decoders import get_decoder




class DAInverseNet(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg_list, sep_decoder_cfg_list):
        super(DAInverseNet, self).__init__()
        self.multi_head = BridgedInverseNet_v2(encoder_cfg=encoder_cfg, decoder_cfg_list=decoder_cfg_list)
        self.sep_endecoder = BridgedInverseNet_v4(encoder_cfg=encoder_cfg, decoder_cfg_list=sep_decoder_cfg_list)
    def forward(self, x, encode_feat=False):
        if not encode_feat:
            ret_multi_head = self.multi_head(x)
            ret_sep = self.sep_endecoder(x)
            return ret_multi_head + ret_sep
        else:
            ret_multi_head, feats_asnl = self.multi_head(x, encode_feat=True)
            ret_sep, feats_c = self.sep_endecoder(x, encode_feat=True)
            feats_dict = {
                'ASNL': feats_asnl,
                'C': feats_c
            }
            return ret_multi_head + ret_sep, feats_dict

class BridgedInverseNet_v2(nn.Module):
    """
    ASNL
    """

    def __init__(self, encoder_cfg, decoder_cfg_list):
        super(BridgedInverseNet_v2 , self).__init__()
        self.encoder = get_encoder(encoder_cfg)
        self.merger = FusionBlock()

        self.decoder_0_ab, self.decoder_0_c = get_decoder(1027, decoder_cfg_list[0], decoder_cfg_list[0].type)
        self.decoder_0_ab_bridge, _ = get_decoder(1027, decoder_cfg_list[0], decoder_cfg_list[0].type)
        self.decoder_1_ab, self.decoder_1_c = get_decoder(1027, decoder_cfg_list[1], decoder_cfg_list[1].type)
        self.decoder_1_ab_bridge, _ = get_decoder(1027, decoder_cfg_list[1], decoder_cfg_list[1].type)
        self.decoder_2_ab, self.decoder_2_c = get_decoder(1027, decoder_cfg_list[2], decoder_cfg_list[2].type)
        self.decoder_2_ab_bridge, _ = get_decoder(1027, decoder_cfg_list[2], decoder_cfg_list[2].type)
        
        self.decoder_3 = get_decoder(1027, decoder_cfg_list[3], decoder_cfg_list[3].type)
        self.decoder_3_bridge = get_decoder(1027, decoder_cfg_list[3], decoder_cfg_list[3].type)
    
        self.W_0 = nn.Sequential(
            nn.Conv2d(in_channels=decoder_cfg_list[0].intermediate_dim,
                      out_channels=decoder_cfg_list[0].intermediate_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_cfg_list[0].intermediate_dim)
        )
        nn.init.constant_(self.W_0[1].weight, 0)
        nn.init.constant_(self.W_0[1].bias, 0)

        self.W_1 = nn.Sequential(
            nn.Conv2d(in_channels=decoder_cfg_list[1].intermediate_dim,
                      out_channels=decoder_cfg_list[1].intermediate_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_cfg_list[1].intermediate_dim)
        )
        nn.init.constant_(self.W_1[1].weight, 0)
        nn.init.constant_(self.W_1[1].bias, 0)

        self.W_2 = nn.Sequential(
            nn.Conv2d(in_channels=decoder_cfg_list[2].intermediate_dim,
                      out_channels=decoder_cfg_list[2].intermediate_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_cfg_list[2].intermediate_dim)
        )
        nn.init.constant_(self.W_2[1].weight, 0)
        nn.init.constant_(self.W_2[1].bias, 0)

        self.W_3 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=1, kernel_size=1),
            nn.BatchNorm1d(1)
        )
        nn.init.constant_(self.W_3[1].weight, 0)
        nn.init.constant_(self.W_3[1].bias, 0)

    def forward(self, x, encode_feat=False, fac=1):
        feat_en = self.encoder(x)
        merg = self.merger(feat_en)

        src_0 = self.decoder_0_ab(merg)
        bridge_0 = self.decoder_0_ab_bridge(merg)
        residual_bridged_feats_0 = src_0 + fac * self.W_0(bridge_0)

        src_1 = self.decoder_1_ab(merg)
        bridge_1 = self.decoder_1_ab_bridge(merg)
        residual_bridged_feats_1 = src_1 + fac * self.W_1(bridge_1)

        src_2 = self.decoder_2_ab(merg)
        bridge_2 = self.decoder_2_ab_bridge(merg)
        residual_bridged_feats_2 = src_2 + fac * self.W_2(bridge_2)

        src_3 = self.decoder_3(merg).unsqueeze(dim=1)
        bridge_3 = self.decoder_3_bridge(merg).unsqueeze(dim=1)
        residual_bridged_feats_3 = src_3 + fac * self.W_3(bridge_3)

        ret = [
            self.decoder_0_c(residual_bridged_feats_0),
            self.decoder_1_c(residual_bridged_feats_1),
            self.decoder_2_c(residual_bridged_feats_2),
            residual_bridged_feats_3.squeeze(dim=1)
        ]

        feats = [
            bridge_0,
            bridge_1,
            bridge_2,
            bridge_3,
        ]

        if encode_feat:
            return ret, feats
        else:
            return ret

    @torch.no_grad()
    def get_feature(self, x):
        return self.encoder(x)

class BridgedInverseNet_v4(nn.Module):
    """
    C only
    """

    def __init__(self, encoder_cfg, decoder_cfg_list):
        super(BridgedInverseNet_v4 , self).__init__()
        self.encoder = get_encoder(encoder_cfg)
        self.merger = FusionBlock()

        self.decoder_0_ab, self.decoder_0_c = get_decoder(1027, decoder_cfg_list[0], decoder_cfg_list[0].type)
        self.decoder_0_ab_bridge, _ = get_decoder(1027, decoder_cfg_list[0], decoder_cfg_list[0].type)
    
        self.W_0 = nn.Sequential(
            nn.Conv2d(in_channels=decoder_cfg_list[0].intermediate_dim,
                      out_channels=decoder_cfg_list[0].intermediate_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_cfg_list[0].intermediate_dim)
        )
        nn.init.constant_(self.W_0[1].weight, 0)
        nn.init.constant_(self.W_0[1].bias, 0)

    def forward(self, x, encode_feat=False, fac=1):
        feat_en = self.encoder(x)
        merg = self.merger(feat_en)

        src_0 = self.decoder_0_ab(merg)
        bridge_0 = self.decoder_0_ab_bridge(merg)
        residual_bridged_feats_0 = src_0 + fac * self.W_0(bridge_0)

        ret = [
            self.decoder_0_c(residual_bridged_feats_0)
        ]

        feats = [
            bridge_0
        ]

        if encode_feat:
            return ret, feats
        else:
            return ret

    @torch.no_grad()
    def get_feature(self, x):
        return self.encoder(x)


class MultiHeadGenerator(nn.Module):

    def __init__(self, encoder_cfg, decoder_cfg_list):
        super(MultiHeadGenerator, self).__init__()
        self.encoder = get_encoder(encoder_cfg)
        self.merger = FusionBlock()
        self.decoder = MultiHeadDecoder([get_decoder(1027, cfg, cfg.type) for cfg in decoder_cfg_list])

    def forward(self, x, encode_feat=False, decode_feat=False):
        feat_en = self.encoder(x)
        merg = self.merger(feat_en)
        ret = self.decoder(merg, with_feat=decode_feat)

        if encode_feat:
            return ret, feat_en
        else:
            return ret

    @torch.no_grad()
    def get_feature(self, x):
        return self.encoder(x)

class MultiHeadDecoder(nn.Module):
    def __init__(self, decoders: list):
        super().__init__()
        self.decoders = nn.ModuleList(decoders)

    def forward(self, x, with_feat=False):
        if isinstance(x, dict):
            ret = []
            for i, decoder in enumerate(self.decoders):
                x['num'] = i
                ret.append(decoder(x, with_feat))
            return ret
        else:
            return [decoder(x, with_feat) for decoder in self.decoders]