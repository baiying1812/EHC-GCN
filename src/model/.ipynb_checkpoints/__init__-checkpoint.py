import math
from torch import nn

from . import layers
from .nets import EfficientGCN
from .activations import *


__activations = {
    'relu': nn.ReLU(inplace=True),
    'relu6': nn.ReLU6(inplace=True),
    'hswish': HardSwish(inplace=True),
    'swish': Swish(inplace=True),
}
##复合缩放策略
def rescale_block(block_args, scale_args, scale_factor):##([[48,1,0.5],[24,1,0.5],[64,2,1],[128,2,1]], [1.2,1.35], 0)
    channel_scaler = math.pow(scale_args[0], scale_factor)##math.pow([1.2], 0) pow：幂运算
    depth_scaler = math.pow(scale_args[1], scale_factor)##math.pow([1.35], 0)
    new_block_args = []
    for [channel, stride, depth] in block_args:##[[48,1,0.5], [24,1,0.5], [64,2,1], [128,2,1]]
        channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)##round：四舍五入
        depth = int(round(depth * depth_scaler))
        new_block_args.append([channel, stride, depth])##[48,1,0.5],[32,1,0.5],[64，2,1],[128，2,1]
    return new_block_args

def create(model_type, act_type, block_args, scale_args, **kwargs):
    ##看yaml文件(EfficientGCN-B0, swish, [[48,1,0.5],[24,1,0.5],[64,2,1],[128,2,1]], [1.2,1.35])
    kwargs.update({
        'act': __activations[act_type],##HardSwish
        'block_args': rescale_block(block_args, scale_args, int(model_type[-1])),##类似[48,1,0.5],[32,1,0.5],[],[]
    })
    return EfficientGCN(**kwargs)
