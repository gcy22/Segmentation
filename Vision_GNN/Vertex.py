# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from Vision_GNN.base_layer import BasicConv, batched_index_select, act_layer
from Vision_GNN.edge import DenseDilatedKnnGraph
from Vision_GNN.position_embedding import get_3d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from model.CNN_parts import DepthWiseConv3d

class MRConv3d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', kernel_size=None, norm=None, bias=True, dilation=1):
        super(MRConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias, dilation=dilation, conv_type="3d")

    def forward(self, x, edge_index, y=None):
        '''
        x: [B, C, D, H, W]  (Batch, Channels, Depth, Height, Width)
        '''
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])

        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, d, h, w = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, d, h, w)
        return self.nn(x)


class MRConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, K=1):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.K = K
        self.mean = 0
        self.std = 0

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_j = x - x

        x_rolled = torch.cat([x[:, :, -D//2:, :, :], x[:, :, :-D//2, :, :]], dim=2)
        x_rolled = torch.cat([x_rolled[:, :, :, -H//2:, :], x_rolled[:, :, :, :-H//2, :]], dim=3)
        x_rolled = torch.cat([x_rolled[:, :, :, :, -W//2:], x_rolled[:, :, :, :, :-W//2]], dim=4)

        norm = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)

        self.mean = torch.mean(norm, dim=[2, 3, 4], keepdim=True)
        self.std = torch.std(norm, dim=[2, 3, 4], keepdim=True)

        for i in range(0, D, self.K):
            x_rolled = torch.cat([x[:, :, -i:, :, :], x[:, :, :-i, :, :]], dim=2)
            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)
            mask = torch.where(dist < self.mean - self.std, 1, 0)
            x_rolled_and_masked = (x_rolled - x) * mask
            x_j = torch.max(x_j, x_rolled_and_masked)

        for j in range(0, H, self.K):
            x_rolled = torch.cat([x[:, :, :, -j:, :], x[:, :, :, :-j, :]], dim=3)
            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)
            mask = torch.where(dist < self.mean - self.std, 1, 0)
            x_rolled_and_masked = (x_rolled - x) * mask
            x_j = torch.max(x_j, x_rolled_and_masked)

        for k in range(0, W, self.K):
            x_rolled = torch.cat([x[:, :, :, :, -k:], x[:, :, :, :, :-k]], dim=4)
            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)
            mask = torch.where(dist < self.mean - self.std, 1, 0)
            x_rolled_and_masked = (x_rolled - x) * mask
            x_j = torch.max(x_j, x_rolled_and_masked)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class EdgeConv3d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv3d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias, conv_type="3d")

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE3D(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE3D, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias, conv_type="3d")
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias, conv_type="3d")

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv3d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv3d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias, conv_type="3d")
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv3d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv3d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv3d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv3d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE3D(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv3d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError(f'conv:{conv} is not supported')

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv3d(GraphConv3d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv3d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, D, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool3d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv3d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, D, H, W).contiguous()


class Grapher3D(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher3D, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            DepthWiseConv3d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels),
        )
        self.graph_conv = DyGraphConv3d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            DepthWiseConv3d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(get_3d_relative_pos_embed(in_channels,
                int(n ** (1/3)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r * r)), mode='trilinear', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, D, H, W):
        if relative_pos is None or D * H * W == self.n:
            return relative_pos
        else:
            N = D * H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="trilinear").squeeze(0)

    def forward(self, x):
        """
        x: (B, C, D, H, W) -> (B, C, D, H, W)
        """
        _tmp = x
        x = self.fc1(x)
        B, C, D, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, D, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)

        x = self.drop_path(x) + _tmp
        return x
