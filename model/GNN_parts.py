
# from misc import DropPath2D, Identity, trunc_array
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from Vision_GNN.Vertex import MRConv3d, MRConv4d
from Vision_GNN.edge import DenseDilatedKnnGraph
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Vision_GNN.Vertex import DepthWiseConv3d, DyGraphConv3d
from Vision_GNN.position_embedding import get_3d_relative_pos_embed

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
        _tmp = x
        x = self.fc1(x)
        B, C, D, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, D, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)

        x = self.drop_path(x) + _tmp
        return x

class ConditionalPositionEncoding(nn.Module):
    """
    Implementation of conditional positional encoding. For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            bias=True, groups=in_channels)

    def forward(self, x):
        x = self.pe(x) + x
        return x

class Grapher(nn.Module):
    def __init__(self, in_channels, drop_path=0.0, K=2, dilation=1):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K
        self.dilation = dilation

        self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)

        self.fc2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm3d(in_channels),
        )  # out_channels back to 1x

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        return x

class FFN3D(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop_path=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.fc1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv3d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        # print(f"Shape after Grapher before FFN: {x.shape}")
        x = self.fc1(x)
        x = self.fc2(x)
        x = shortcut + self.drop_path(x)
        # print(f"Shape after FFN: {x.shape}")
        return x




class ViG_Block3D(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., dilation=1, kernel_size=3, stride=1):
        super().__init__()
        self.grapher = Grapher(dim, drop_path=drop_path, dilation=dilation)     # Grapher代替Self-Attention
        hidden_channels = int(dim * mlp_ratio)
        # self.mlp = ConvFFN(in_channels=dim, hidden_channels=hidden_channels, kernel_size=kernel_size, stride=stride, out_channels=dim, drop_path=drop_path, )
        self.mlp = FFN3D(in_channels=dim, hidden_channels=int(dim * mlp_ratio), out_channels=dim, drop_path=drop_path)
    def forward(self, x):
        x = self.grapher(x)
        x = self.mlp(x)
        # print(x.shape)
        return x