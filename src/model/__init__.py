import math
from torch import nn

from . import layers
from .nets import EHC_GCN
from .activations import *


__activations = {
    'relu': nn.ReLU(inplace=True),
    'relu6': nn.ReLU6(inplace=True),
    'hswish': HardSwish(inplace=True),
    'swish': Swish(inplace=True),
}

def create(act_type, **kwargs):
    kwargs.update({
        'act': __activations[act_type],
    })
    return EHC_GCN(**kwargs)
