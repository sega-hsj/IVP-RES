#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .my_Vis_Head import SingleHead


def kernel_conv(kernel,encode_feat):
    B,C,H,W=encode_feat.shape
    encode_feat=encode_feat.reshape(B,C,H*W)
    kernel=kernel.reshape(B,1,C)
    pred=torch.matmul(kernel,encode_feat) #B,1,HW
    pred=pred.reshape(B,1,H,W)
    return pred

class Thres_Kernel_Head(nn.Module):
    def __init__(self, cfg, in_dim=256, out_dim=64, thres=0.4):
        super(Thres_Kernel_Head, self).__init__()
        norm = cfg.MODEL.POSITION_HEAD.NORM
        self.thres = thres
        self.kernel_trans = nn.Conv2d(256, 64, kernel_size=1)

    def generate_kernel(self,pred_region,kernel_weight):
        C,H,W=kernel_weight.shape
        kernel_weight=kernel_weight.reshape(C,-1) #C,HW
        pred_region=pred_region.reshape(-1) # 1,HW
        
        keep=pred_region > self.thres
        if keep.sum()==0:
            return torch.zeros(64,1,1).cuda()
        kernel=kernel_weight[:,keep] # C,num
        kernel=kernel.mean(dim=-1) # C
        kernel=kernel.reshape(1,C,1,1)
        kernel=self.kernel_trans(kernel).squeeze(0)
        return kernel 

    def forward(self,pred_region,kernel_weight,encode_feat):
        B,C,H,W=encode_feat.shape
        kernels=[]
        for i in range(B):
            kernel=self.generate_kernel(pred_region[i],kernel_weight[i]) #1,H,W  C,H,W  = 1,C,1,1
            kernels.append(kernel)
        kernel=torch.stack(kernels)
        preds=kernel_conv(kernel,encode_feat)
        return preds

class Dynamic_Kernel_Head(nn.Module):
    def __init__(self, cfg, feat_dim=256, kernel_dim=256):
        super(Dynamic_Kernel_Head, self).__init__()
        norm = cfg.MODEL.POSITION_HEAD.NORM

        self.dynamic_conv = SingleHead(feat_dim+kernel_dim, 256, 2, kernel_size=1, padding=0, deform=False, coord=False, norm=norm, name='dynamic_conv')
        self.kernel_trans = nn.Conv2d(256, 64, kernel_size=1)

    def generate_kernel(self,vis_feat,kernel_weight):
        feat=torch.cat([vis_feat,kernel_weight],dim=1)
        kernel_feat=self.dynamic_conv(feat)
        B,C=kernel_feat.shape[:2]
        kernel=kernel_feat.reshape(B,C,-1).mean(dim=-1).reshape(B,C,1,1)
        return kernel

    def forward(self,vis_feats,kernel_weights,encode_feat):
        kernels=[]
        for vis_feat,kernel_weight in zip(vis_feats,kernel_weights):
            kernel=self.generate_kernel(vis_feat,kernel_weight)
            kernel=self.kernel_trans(kernel)
            kernels.append(kernel)
        kernels=torch.stack(kernels)
        kernel=kernels.mean(dim=0)
        pred=kernel_conv(kernel,encode_feat)
        return pred

class Softmax_Kernel_Head(nn.Module):
    def __init__(self, cfg, input_dim=256, output_dim=64):
        super(Softmax_Kernel_Head, self).__init__()
        norm = cfg.MODEL.POSITION_HEAD.NORM
        self.kernel_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def generate_kernel(self,score,kernel_weight):
        B,C,H,W=score.shape
        score=score.reshape(B,1,-1)
        score=torch.softmax(score,dim=-1) #B,1,HW

        B,C,H,W=kernel_weight.shape
        kernel_weight=kernel_weight.reshape(B,C,-1) #B,C,HW
        score=score.repeat(1,C,1) #B,C,HW

        kernel=kernel_weight*score
        kernel=kernel.sum(dim=-1).reshape(B,C,1,1)
        return kernel

    def forward(self,soft_scores,kernel_weights,encode_feat):
        kernels=[]
        for soft_score,kernel_weight in zip(soft_scores,kernel_weights):
            kernel=self.generate_kernel(soft_score,kernel_weight)
            kernel=self.kernel_trans(kernel)
            kernels.append(kernel) #B,C,1,1
        kernels=torch.stack(kernels)
        kernel=kernels.mean(dim=0)
        pred=kernel_conv(kernel,encode_feat)
        return pred

class Softmax_Inst_Head(nn.Module):
    def __init__(self, cfg, input_dim=256, output_dim=64):
        super(Softmax_Inst_Head, self).__init__()
        norm = cfg.MODEL.POSITION_HEAD.NORM
        self.kernel_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def generate_kernel(self,score,kernel_weight):
        B,C,H,W=score.shape
        score=score.reshape(B,1,-1)
        score=torch.softmax(score,dim=-1) #B,1,HW

        B,C,H,W=kernel_weight.shape
        kernel_weight=kernel_weight.reshape(B,C,-1) #B,C,HW
        score=score.repeat(1,C,1) #B,C,HW

        kernel=kernel_weight*score
        kernel=kernel.sum(dim=-1).reshape(B,C,1,1)
        return kernel

    def forward(self,soft_score,kernel_weight,encode_feat):
        kernel=self.generate_kernel(soft_score,kernel_weight)
        kernel=self.kernel_trans(kernel) #B,64,1,1
        B,C,H,W=encode_feat.shape
        kernel=kernel.repeat(1,1,H,W)
        inst_feat=kernel*encode_feat
        return inst_feat


class Sigmoid_Kernel_Head(nn.Module):
    def __init__(self, cfg, input_dim=256, output_dim=64):
        super(Sigmoid_Kernel_Head, self).__init__()
        norm = cfg.MODEL.POSITION_HEAD.NORM
        self.kernel_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def generate_kernel(self,score,kernel_weight):
        B,C,H,W=kernel_weight.shape

        score=score.sigmoid() #B,1,H,W
        score_sum=score.sum()
        score=score.repeat(1,C,1,1) #B,C,H,W
        kernel=kernel_weight*score #B,C,H,W
        kernel=kernel.reshape(B,C,-1) #B,C,HW
        kernel=kernel.sum(dim=-1) #B,C
        kernel=kernel/score_sum
        kernel=kernel.reshape(B,C,1,1)
        return kernel

    def forward(self,soft_scores,kernel_weights,encode_feat):
        kernels=[]
        for soft_score,kernel_weight in zip(soft_scores,kernel_weights):
            kernel=self.generate_kernel(soft_score,kernel_weight)
            kernel=self.kernel_trans(kernel)
            kernels.append(kernel) #B,C,1,1
        kernels=torch.stack(kernels)
        kernel=kernels.mean(dim=0)
        pred=kernel_conv(kernel,encode_feat)
        return pred

if __name__=="__main__":
    exit(0)
