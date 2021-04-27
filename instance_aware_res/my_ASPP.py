#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .my_Vis_Head import SingleHead

class ASPP_Head(nn.Module):
    def __init__(self, cfg, input_shape=512, dilat=[6,12,18]):
        super(ASPP_Head, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(input_shape, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.GroupNorm(32, 256)

        self.conv_3x3_1 = nn.Conv2d(input_shape, 256, kernel_size=3, stride=1, padding=dilat[0], dilation=dilat[0])
        self.bn_conv_3x3_1 = nn.GroupNorm(32, 256)

        self.conv_3x3_2 = nn.Conv2d(input_shape, 256, kernel_size=3, stride=1, padding=dilat[1], dilation=dilat[1])
        self.bn_conv_3x3_2 = nn.GroupNorm(32, 256)

        self.conv_3x3_3 = nn.Conv2d(input_shape, 256, kernel_size=3, stride=1, padding=dilat[2], dilation=dilat[2])
        self.bn_conv_3x3_3 = nn.GroupNorm(32, 256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(input_shape, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.GroupNorm(32, 256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.GroupNorm(32, 256)


    def forward(self, feature_map, lang_feat=None):
        #input :B, C, H,W
        #output:B,256,H,W
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out