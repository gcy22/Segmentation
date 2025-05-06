import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
from timm.models.helpers import named_apply
from functools import partial


class DepthWiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        out_channels = (out_channels // in_channels) * in_channels
        self.depth_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                    stride=stride, dilation=dilation, groups=in_channels,
                                    bias=False)  # groups用于将输入通道和输出通道分组，每组独立进行卷积操作

        num_groups = max(1, in_channels // 4)
        self.norm_layer = nn.GroupNorm(num_groups, out_channels)

        self.point_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)

        self.activation = nn.GELU()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        out = self.depth_conv(x)
        # print(f"Shape after depth_conv: {x.shape}")
        # out = self.norm_layer(out)
        out = self.point_conv(out)
        # print(f"Shape after point_conv: {x.shape}")
        return out


# ---------Before Stem : 1*1*48*256*256 ; After Stem : 1*32*24*128*128--------- #
class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            DepthWiseConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),  # after BatchNorm3d 1*32*48*256*256

            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            # after BatchNorm3d 1*32*24*128*128
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # 打印输入形状
        x = self.stem[0](x)  # DepthWiseConv3d
        x = self.stem[1](x)  # BatchNorm3d
        # print(f"After first BatchNorm3d: {x.shape}")  # 打印BatchNorm3d后的形状
        x = self.stem[2](x)  # ReLU
        x = self.stem[3](x)  # Conv3d
        x = self.stem[4](x)  # 第二个BatchNorm3d
        # print(f"After second BatchNorm3d: {x.shape}")  # 打印第二个BatchNorm3d后的形状
        x = self.stem[5](x)  # ReLU
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            DepthWiseConv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  # 双线性插值上采样
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):  # x2是来自编码器的跳跃连接
        x1 = self.up(x1)
        # input is CDHW
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        # print(f"x1 shape before pad: {x1.shape}")
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])

        # mask = torch.ones_like(x1)
        # x2 = self.gab(x2, x1, mask)
        # print(f"x2 shape before concat: {x2.shape}, x1 shape before concat: {x1.shape}")
        x = torch.cat([x2, x1], dim=1)
        # print(f"x shape after concat: {x.shape}")

        return self.relu(self.batchnorm(self.conv(x)))


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


# F_g表示输入特征图 g 的channels数。 F_l输入特征图 x 的channels数, F_int：g 和 x 经过卷积操作后的中间特征的通道数。
class GroupedAttention3D(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(GroupedAttention3D, self).__init__()

        if kernel_size == 1:
            groups = 1

        # 使用 3D 卷积
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        # 激活函数
        self.activation = act_layer(activation, inplace=True)

        # 初始化权重
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)  # 处理 g
        x1 = self.W_x(x)  # 处理 x
        # 调整 g1 的大小以匹配 x1
        if g1.shape[2:] != x1.shape[2:]:
            upsample = nn.Upsample(size=x1.shape[2:], mode='trilinear', align_corners=True)
            g1 = upsample(g1)

        psi = self.activation(g1 + x1)  # 激活函数
        psi = self.psi(psi)  # 生成门控权重

        return x * psi  # 应用门控权重


def _init_weights(module, name, scheme=''):
    # 处理 2D 和 3D 卷积层
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # EfficientNet 类似的初始化方法
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            elif isinstance(module, nn.Conv3d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # 处理 2D 和 3D 批归一化层
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    # 处理层归一化层
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class Upsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample3D, self).__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=2,  # 2x2x2 卷积核
            stride=2,  # 步长 2，实现 2 倍上采样
            padding=0
        )

    def forward(self, x):
        # print(f"Before up shape: {x.shape}")
        x = self.upsample(x)
        # print(f"after up shape: {x.shape}")
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding='same'),
        )

    def forward(self, x):
        # print("before outconv shape of x :", x.size())
        x = self.out(x)
        x = x.view(1, 2, *x.shape[2:])
        # print("finally outconv shape of x :", x.size())
        return x