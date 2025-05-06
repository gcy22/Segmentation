# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# modified from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils_module
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# relative position embedding
# References: https://arxiv.org/abs/2009.13658
# --------------------------------------------------------

def get_3d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width and depth
    return:
    pos_embed: [grid_size*grid_size*grid_size, grid_size*grid_size*grid_size]相对位置嵌入矩阵
    """
    pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    :param embed_dim: 嵌入维度（通常是偶数）
    :param grid_size: 网格的大小（例如，体素的尺寸）
    :param cls_token: 是否包含一个分类 token
    :return: 生成的三维位置嵌入
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_d = np.arange(grid_size, dtype=np.float32)

    grid_h, grid_w, grid_d = np.meshgrid(grid_h, grid_w, grid_d)  # 生成三维网格
    grid = np.stack([grid_h, grid_w, grid_d], axis=0)

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    :param embed_dim: 嵌入维度
    :param grid: 三维网格
    :return: 生成的嵌入矩阵
    """
    assert embed_dim % 3 == 0

    # 使用一半维度来编码高度（h），宽度（w）和深度（d）
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*D, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*D, D/3)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*D, D/3)

    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1)  # (H*W*D, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb
