#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .my_Vis_Head import SingleHead

class Fusion_Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        norm = cfg.MODEL.POSITION_HEAD.NORM
        
        self.vis_trans1 = SingleHead(2048+2, 1024, 2, kernel_size=3, padding=1, deform=False, coord=True, norm=norm, name='vis_trans1',activation=F.leaky_relu)
        self.lang_trans1 = SingleHead(1024, 1024, 1, kernel_size=1, padding=0, deform=False, coord=False, norm=norm, name='lang_trans1',activation=F.leaky_relu)

        self.vis_trans2 = SingleHead(1024+2, 1024, 2, kernel_size=3, padding=1, deform=False, coord=True, norm=norm, name='vis_trans2',activation=F.leaky_relu)
        self.lang_trans2 = SingleHead(1024, 1024, 1, kernel_size=1, padding=0, deform=False, coord=False, norm=norm, name='lang_trans2',activation=F.leaky_relu)

        self.vis_trans3 = SingleHead(512+2, 1024, 2, kernel_size=3, padding=1, deform=False, coord=True, norm=norm, name='vis_trans3',activation=F.leaky_relu)
        self.lang_trans3 = SingleHead(1024, 1024, 1, kernel_size=1, padding=0, deform=False, coord=False, norm=norm, name='lang_trans3',activation=F.leaky_relu)

        self.fuse_trans1 = SingleHead(1024, 512, 2, kernel_size=3, padding=1, deform=False, coord=False, norm=norm, name='fuse_trans1',activation=F.leaky_relu)
        self.fuse_trans2 = SingleHead(1024+512, 512, 2, kernel_size=3, padding=1, deform=False, coord=False, norm=norm, name='fuse_trans2',activation=F.leaky_relu)
        self.fuse_trans3 = SingleHead(1024+512, 512, 2, kernel_size=3, padding=1, deform=False, coord=False, norm=norm, name='fuse_trans3',activation=F.leaky_relu)


    def mutan(self,vis_feat,lang_feat,vis_trans,lang_trans):
        H,W=vis_feat.shape[-2:]
        vis_feat=vis_trans(vis_feat)
        lang_feat=lang_trans(lang_feat.unsqueeze(-1).unsqueeze(-1)).repeat(1,1,H,W)
        mutan_feat=vis_feat*lang_feat
        return mutan_feat

    def forward(self, vis_feat, lang_feat):
        #input: [res3,res4,res5]
        #ouput: B,512,H,W
        v1=vis_feat['res5'] # 2048
        m1=self.mutan(v1,lang_feat,self.vis_trans1,self.lang_trans1)
        v2=vis_feat['res4'] # 1024
        m2=self.mutan(v2,lang_feat,self.vis_trans2,self.lang_trans2)
        v3=vis_feat['res3'] # 512
        m3=self.mutan(v3,lang_feat,self.vis_trans3,self.lang_trans3)

        H,W=v2.shape[-2:]
        m1u=F.upsample(m1,size=(H,W),mode="bilinear", align_corners=False)
        m1ut=self.fuse_trans1(m1u)
        m2=torch.cat([m1ut,m2],dim=1)

        H,W=v3.shape[-2:]
        m2u=F.upsample(m2,size=(H,W),mode="bilinear", align_corners=False)
        m2ut=self.fuse_trans2(m2u)
        m3=torch.cat([m2ut,m3],dim=1)

        m3t=self.fuse_trans3(m3)

        return m3t
