import torch
from torch import nn


class Basic_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(Basic_Layer, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)

        self.residual = nn.Identity() if residual else Zero_Layer()
        self.act = act

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x


class Spatial_Graph_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, v_size, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, v_size, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )


class Temporal_Basic_Layer(nn.Module):
    def __init__(self, last_channel, channel, temporal_window_size, bias, act, stride=1, residual=True, **kwargs):
        super(Temporal_Basic1_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2d(last_channel, channel, (temporal_window_size,1), (stride,1), (padding,0), bias=bias)
        self.bn = nn.BatchNorm2d(channel)
        self.act = act
        if residual:
            self.residual = nn.Sequential(
                nn.Conv2d(last_channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )
    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x



class Temporal_SG_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = in_channel // reduct_ratio
        self.act = act

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (temporal_window_size,1), 1, (padding,0), groups=in_channel, bias=bias),
            nn.BatchNorm2d(in_channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(in_channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, out_channel, 1, bias=bias),
            nn.BatchNorm2d(out_channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (temporal_window_size,1), stride, (padding,0), groups=out_channel, bias=bias),
            nn.BatchNorm2d(out_channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride, bias=bias),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res


class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, v_size, bias, edge, A, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1##3
        self.gcn = nn.Conv2d(in_channel, out_channel*self.s_kernel_size, 1, bias=bias)##6,64,3
        self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1
        self.v_size = v_size
        self.value = self.A[0][0][0]
        self.diagonal_matrix = nn.Parameter(torch.eye(self.v_size) * self.value)
        self.expanded_tensor = nn.Parameter(torch.stack([self.diagonal_matrix] * self.s_kernel_size, dim=0), requires_grad=False)

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        
        # x = torch.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
        if self.v_size != 0:
            x = torch.einsum('nkctv,kvw->nctw', (x, self.expanded_tensor)).contiguous()
        else:
            x = torch.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
        return x
