import torch
from torch import nn

from .. import utils as U
from .attentions import Attention_Layer
from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer, Temporal_Basic1_Layer


class EfficientGCN(nn.Module):
    '''data_shape：##[3分支JVB, 6, 帧数, 25个关节点, 2个人]
    block_args:[48,1,0.5],[16,1,0.5],[64，2,1],[128，2,1][channel, stride, depth](复合缩放策略EfficientGCN-B0后
    fusion_stage:2
    stem_channel: 64
    '''
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(EfficientGCN, self).__init__()

        num_input, num_channel, T, V, M = data_shape##3,6

        # input branches三个分支，前2=fusion_stage个GCN模块
        self.input_branches0 = nn.ModuleList([EfficientGCN_Blocks(
            init_channel = stem_channel,## 64，
            channel = 48,##[[32,1,0.5]]
            input_channel = num_channel,## 6
            **kwargs
        ) for _ in range(num_input)])##3,创建3个 EfficientGCN_Blocks 实例，并将它们存储在一个 nn.ModuleList 对象中
        
        # last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]##16
        self.input_branches1 = nn.ModuleList([EfficientCNN_Blocks(
            init_channel = V,## 25，
            block_args = [[V,1,1]],##cnn_channel\[[V,1,1]]
            #pre_channel = last_channel,
            **kwargs
        ) for _ in range(num_input)])##3,创建3个 EfficientGCN_Blocks 实例，并将它们存储在一个 nn.ModuleList 对象中

        # main stream
        self.main_stream0 = EfficientGCN_Blocks(
            init_channel = num_input * 48,##16*3
            channel = 64,##[64，2,1],[128，2,1]
            **kwargs
        )
        self.main_stream1 = EfficientCNN_Blocks(
            init_channel = V,## 25，
            block_args = [[V,1,1]],##cnn_channel\[[V,1,1]]
            **kwargs
        )
        self.main_stream2 = EfficientGCN_Blocks(
            init_channel = 64,##16*3
            channel = 128,##[64，2,1],[128，2,1]
            **kwargs
        )
        self.main_stream3 = EfficientCNN_Blocks(
            init_channel = V,## 25，
            block_args = [[V,1,1]],##cnn_channel\[[V,1,1]]
            **kwargs
        )

        # output
        #last_channel = num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]##128
        self.classifier = EfficientGCN_Classifier(128, **kwargs)

        # init parameters
        init_param(self.modules())

    def forward(self, x):

        N, I, C, T, V, M = x.size()##batchsize,3个分支input，6_channel，帧数，25个关节点，每帧最多选择2个人
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V)
        ##x ：3个输入分支，batchsize*2个人，6_channel，帧数，25个关节点

        # input branches
        x = torch.cat([branch(x[i]).unsqueeze(0) for i, branch in enumerate(self.input_branches0)], dim=0)
        x = x.permute(0, 1, 4, 3, 2)
        #print(x.size())
        x = torch.cat([branch(x[i]) for i, branch in enumerate(self.input_branches1)], dim=3)
        ##x ：batchsize*2个人，48=16_channel*3分支，帧数，25个关节点
        ##channel：6 →64 →48 →16 →16*3

        # main stream
        x = x.permute(0, 3, 2, 1)
        x = self.main_stream0(x)
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


class EfficientGCN_Blocks(nn.Sequential):
    ##          (self,   64,  [[48,1,0.5],[16,1,0.5]],  SG,     [5,2],     6
    def __init__(self, init_channel, channel, layer_type, kernel_size, input_channel=0, v_size=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size##5,2

        if input_channel > 0:  # if the blocks in the input branches
            self.add_module('init_bn', nn.BatchNorm2d(input_channel))
            self.add_module('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, v_size,**kwargs))
            self.add_module('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel##64
        temporal_layer = U.import_class(f'src.model.layers.Temporal_{layer_type}_Layer')

        self.add_module(f'block_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, v_size, **kwargs))
        self.add_module(f'block_att', Attention_Layer(channel, **kwargs))

class EfficientCNN_Blocks(nn.Sequential):
    #def __init__(self, init_channel, block_args, layer_type, kernel_size, **kwargs):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, att_type='stja1', **kwargs):
        super(EfficientCNN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size##5,2
        temporal_layer = U.import_class(f'src.model.layers.Temporal_{layer_type}_Layer')
        last_channel = init_channel
        for i, [channel, stride, depth] in enumerate(block_args):##i从0开始
            for j in range(int(depth)):#
                s = stride if j == 0 else 1
                inchannel=last_channel if j == 0 else channel
                self.add_module(f'block-{i}_tcn-{j}', temporal_layer(inchannel, channel, temporal_window_size, stride=s, **kwargs))
            #self.add_module(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            #self.add_module(f'block-{i}_att', Attention_Layer(pre_channel, att_type, **kwargs))
            last_channel = channel
            

class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(EfficientGCN_Classifier, self).__init__()

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
