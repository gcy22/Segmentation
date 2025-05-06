import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from model.CNN_parts import Stem, Down, UP, OutConv, Upsample3D, GroupedAttention3D
from model.GNN_parts import ViG_Block3D, Grapher
from Vision_GNN.Vertex import Grapher3D

# sys.path.append('/home/tom/fsas/my_ViG')


class Med_Seg_ViG(nn.Module):
    def __init__(self, in_channels, out_channels, training=True, k=9, drop_path=0., dilation=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = Stem(in_channels, 32)
        self.grapher_ffn1 = ViG_Block3D(32, drop_path=drop_path, dilation=dilation)  # After Encoder1 1*32*24*128*128

        self.encoder2 = Down(32, 64)
        self.grapher_ffn2 = ViG_Block3D(64, drop_path=drop_path, dilation=dilation)  # After Encoder2 1*64*12*64*64

        self.encoder3 = Down(64, 128)
        self.grapher_ffn3 = ViG_Block3D(128, drop_path=drop_path, dilation=dilation)  # After Encoder3 1*128*6*32*32

        self.encoder4 = Down(128, 256)  # After Encoder4(Only Downsample) 1*256*3*16*16

        self.grapher = Grapher(256, drop_path=drop_path, dilation=dilation)  # After grapher 1*256*3*16*16

        self.decoder1 = UP(256 + 256, 128)  # concat x1/x2 both 1*256*3*16*16 so neeed channel 256+256
        self.grapher_ffn4 = ViG_Block3D(128, drop_path=drop_path, dilation=dilation)

        self.decoder2 = UP(128 + 128, 64)
        self.grapher_ffn5 = ViG_Block3D(64, drop_path=drop_path, dilation=dilation)

        self.decoder3 = UP(64 + 64, 32)
        # self.en_GA2 = nn.Sequential(Grouped_Attention(128,64, drop_path=drop_path, dilation=dilation))
        self.grapher_ffn6 = ViG_Block3D(32, drop_path=drop_path, dilation=dilation)

        self.up1 = Upsample3D(32, 32)
        self.out_conv = OutConv(32, 2)

        self.GA1 = GroupedAttention3D(F_g=256, F_l=256, F_int=128)
        self.GA2 = GroupedAttention3D(F_g=128, F_l=128, F_int=64)
        self.GA3 = GroupedAttention3D(F_g=64, F_l=64, F_int=32)

    def forward(self, x):
        # Encoder and Graph Attention Modules
        x1 = self.encoder1(x)  # after Stem 1*32*24*128*128
        x1 = self.grapher_ffn1(x1)  # after VIG 1*32*24*128*128

        x2 = self.encoder2(x1)  # after Down1 1*64*12*64*64
        x2 = self.grapher_ffn2(x2)

        x3 = self.encoder3(x2)  # after Down2 1*128*6*32*32
        x3 = self.grapher_ffn3(x3)

        x4 = self.encoder4(x3)  # after Down3 1*256*3*16*16

        x5 = self.grapher(x4)  # after Grapher 1*256*3*16*16

        # Decoder with attention blocks and resizing for skip connections
        # x5 = self.up1(x5)
        x = self.GA1(x5, x4)
        x = self.decoder1(x, x4)  # after up1 1*128*3*16*16
        x = self.grapher_ffn4(x)

        x = self.GA2(x, x3)
        x = self.decoder2(x, x3)  # after up2 1*64*6*32*32
        x = self.grapher_ffn5(x)

        x = self.GA3(x, x2)
        x = self.decoder3(x, x2)  # after up3 1*32*12*64*64
        x = self.grapher_ffn6(x)

        x = self.up1(x)
        outputs = self.out_conv(x)  # after outconv 1*2*24*128*128
        # print("x最终shape", outputs.shape)
        return outputs

