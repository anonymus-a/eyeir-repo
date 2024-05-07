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



def get_norm_layer_by_str(norm: str, norm_dim: int):
    if norm == 'bn': 
        norm = nn.BatchNorm2d(norm_dim)
    elif norm == 'in':  
        norm = nn.InstanceNorm2d(norm_dim)
    elif norm == 'ln': 
        norm = LayerNorm(norm_dim)
    elif norm == 'adain':
        norm = AdaptiveInstanceNorm2d(norm_dim)
    elif norm == 'none':
        norm = None
    else:
        raise NotImplementedError
    return norm

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

def get_activation_by_str(activation: str):
    # initialize activation
    if activation == 'relu':
        activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        activation = nn.PReLU()
    elif activation == 'selu':
        activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'softmax':
        activation = nn.LogSoftmax(dim=1)
    elif activation == 'none':
        activation = None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
    return activation

def get_pad_by_str(pad_type: str, padding: int):
    if pad_type == 'reflect':
        pad = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
        pad = nn.ZeroPad2d(padding)
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)
    return pad

def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.dilation = dilation
        self.padding = padding

        self.pad = get_pad_by_str(pad_type, padding)

        norm_dim = output_dim
        self.norm = get_norm_layer_by_str(norm, norm_dim)
        self.activation = get_activation_by_str(activation)

        if isinstance(self.padding, str) and self.padding.lower() == 'same':
            self.conv = Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=self.dilation)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias,
                                  dilation=self.dilation)

    def forward(self, x):
        if isinstance(self.padding, str) and self.padding.lower() == 'same':
            x = self.conv(x)
        else:
            x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class FusionBlock(nn.Module):
    def __init__(self, resize=None):

        super(FusionBlock, self).__init__()

        self.resize = resize

    def _fuse_feature(self, x, feature):
        if self.resize is None:
            self.resize = x.shape[2:]
        h, w = self.resize
        feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([x, feature], dim=1)
        return x

    def forward(self, features):
        if type(features) is list:
            x = features[0]
            for ii in range(1, len(features)):
                x = self._fuse_feature(x, features[ii])
        elif type(features) is dict:
            x = features['input']
            x = self._fuse_feature(x, features['shallow'])
            x = self._fuse_feature(x, features['low'])
            x = self._fuse_feature(x, features['mid'])
            x = self._fuse_feature(x, features['deep'])
            x = self._fuse_feature(x, features['out'])

        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm='in', act='lrelu', se_reduction=None, res_scale=1):

        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2dBlock(channels, channels, kernel_size=3, stride=1, padding='SAME', dilation=dilation,
                                 norm=norm, activation=act)
        self.conv2 = Conv2dBlock(channels, channels, kernel_size=3, stride=1, padding='SAME', dilation=dilation,
                                 norm=norm, activation='none')

        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y

class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat(
            [F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))