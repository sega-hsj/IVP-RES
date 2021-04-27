#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import Conv2d, get_norm
from .deform_conv_with_off import ModulatedDeformConvWithOff


class SingleHead(nn.Module):
    """
    Build single head with convolutions and coord conv.
    """
    def __init__(self, in_channel, conv_dims, num_convs, deform=False, coord=False, norm='', name='',kernel_size=3,padding=1,activation=F.relu):
        super().__init__()
        self.coord = coord
        self.conv_norm_relus = []
        if deform:
            conv_module = ModulatedDeformConvWithOff
        else:
            conv_module = Conv2d
        for k in range(num_convs):
            conv = conv_module(
                    in_channel if k==0 else conv_dims,
                    conv_dims,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=activation,
                )
            self.add_module("{}_head_{}".format(name, k + 1), conv)
            self.conv_norm_relus.append(conv)

    def forward(self, x):
        if self.coord:
            x = self.coord_conv(x)
        for layer in self.conv_norm_relus:
            x = layer(x)
        return x
    
    def coord_conv(self, feat):
        with torch.no_grad():
            x_pos = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
            y_pos = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
            grid_x, grid_y = torch.meshgrid(x_pos, y_pos)
            grid_x = grid_x.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1)
            grid_y = grid_y.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1)
        feat = torch.cat([feat, grid_x, grid_y], dim=1)
        return feat


class KernelHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.FPN.OUT_CHANNELS
        conv_dims       = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        num_convs       = cfg.MODEL.KERNEL_HEAD.NUM_CONVS
        deform          = cfg.MODEL.KERNEL_HEAD.DEFORM
        coord           = cfg.MODEL.KERNEL_HEAD.COORD
        norm            = cfg.MODEL.KERNEL_HEAD.NORM

        self.kernel_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims,
                                      num_convs,
                                      deform=deform,
                                      coord=coord,
                                      norm=norm,
                                      name='kernel_head')
        self.out_conv = Conv2d(conv_dims, conv_dims, kernel_size=3, padding=1)
        nn.init.normal_(self.out_conv.weight, mean=0, std=0.01)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)
       
    def forward(self, feat):
        x = self.kernel_head(feat)
        x = self.out_conv(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.SEMANTIC_FPN.CONVS_DIM
        conv_dims       = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        num_convs       = cfg.MODEL.FEATURE_ENCODER.NUM_CONVS
        deform          = cfg.MODEL.FEATURE_ENCODER.DEFORM
        coord           = cfg.MODEL.FEATURE_ENCODER.COORD
        norm            = cfg.MODEL.FEATURE_ENCODER.NORM
        
        self.encode_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims, 
                                      num_convs, 
                                      deform=deform,
                                      coord=coord,
                                      norm=norm, 
                                      name='encode_head')

    def forward(self, feat):
        feat = self.encode_head(feat)
        return feat
