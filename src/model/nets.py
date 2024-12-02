import torch
from torch import nn

from .. import utils as U
from .attentions import Attention_Layer
from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer, Temporal_Basic1_Layer


class EHC_GCN(nn.Module):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(EHC_GCN, self).__init__()

        num_input, num_channel, T, V, M = data_shape
        temporal_window_size, max_graph_distance = kernel_size##5,2
        temporal_layer = U.import_class(f'src.model.layers.Temporal_{layer_type}_Layer')

        self.SGC_0 = nn.ModuleList([SGC_Blocks(
            init_channel = stem_channel,，
            channel = channel_args[0],
            input_channel = num_channel,
            **kwargs
        ) for _ in range(num_input)])
        self.EC_TC_layers0 = nn.ModuleList([temporal_layer(V, V, temporal_window_size, stride=1, **kwargs) for _ in range(num_input)])
        
        # main stream
        self.SGC_1 = SGC_Blocks(
            init_channel = num_input * channel_args[0],
            channel = channel_args[1],
            **kwargs)
        self.EC_TC_layers1 = temporal_layer(V, V, temporal_window_size, stride=1, **kwargs)
        
        self.SGC_2 = SGC_Blocks(
            init_channel = channel_args[1],
            channel = channel_args[2],
            **kwargs)
        self.EC_TC_layers2 = temporal_layer(V, V, temporal_window_size, stride=1, **kwargs)
        
        # output
        self.classifier = Classifier(channel_args[2], **kwargs)

        # init parameters
        init_param(self.modules())

    def forward(self, x):
        N, I, C, T, V, M = x.size()
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V)

        # input branches
        x = torch.cat([branch(x[i]).unsqueeze(0) for i, branch in enumerate(self.SGC_0)], dim=0)
        x = x.permute(0, 1, 4, 3, 2)
        x = torch.cat([branch(x[i]) for i, branch in enumerate(self.EC_TC_layers0)], dim=3)
        
        # main stream
        x = x.permute(0, 3, 2, 1)
        x = self.SGC_1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.main_stream1(x)
        
        x = x.permute(0, 3, 2, 1)
        x = self.main_stream2(x)
        x = x.permute(0, 3, 2, 1)
        x = self.main_stream3(x)
        x = x.permute(0, 3, 2, 1)
        ##x ：batchsize*2个人，128_channel，帧数，25个关节点

        # output
        _, C, T, V = x.size()
        #_, V, T, C = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        ##（batchsize，128_channel，帧数，25个关节点，2个人）
        out = self.classifier(feature).view(N, -1)
        ##（batchsize，60个分类，帧数，25个关节点，2个人）→（batchsize，60个分类*帧数*25个关节点*2个人）

        return out, feature


class SGC_Blocks(nn.Sequential):
    def __init__(self, init_channel, channel, kernel_size, input_channel=0, v_size=0, **kwargs):
        super(SGC_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size##5,2

        if input_channel > 0:  # if the blocks in the input branches
            self.add_module('init_bn', nn.BatchNorm2d(input_channel))
            self.add_module('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, v_size,**kwargs))
            self.add_module('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        
        self.add_module(f'block_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, v_size, **kwargs))
        self.add_module(f'block_att', Attention_Layer(channel, **kwargs))

class EC_TC_layer(nn.Sequential):
    def __init__(self, channel, layer_type, kernel_size,  **kwargs):
        super(EC_TC_Blocks, self).__init__()

        temporal_window_size, _ = kernel_size##5,2
        temporal_layer = U.import_class(f'src.model.layers.Temporal_{layer_type}_Layer')
        
        self.add_module(f'block-{i}_tcn-{j}', temporal_layer(channel, channel, temporal_window_size, stride=1, **kwargs))
            

class Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(Classifier, self).__init__()

        self.add_module('gap', nn.AdaptiveAvgPool3d(1))
        self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
        self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))##用到了三维卷积


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
