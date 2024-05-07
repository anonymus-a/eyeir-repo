
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng (tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hrnet import get_face_alignment_net, HighResolutionNet
# __all__是一个字符串list，用来定义模块中对于from XXX import *时要对外导出的符号，即要暴露的借口，但它只对import *起作用，对from XXX import XXX不起作用。

__all__ = ['HighResolutionNet', 'get_face_alignment_net'] #
