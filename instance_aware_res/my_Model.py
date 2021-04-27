import torch
from torch import nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .backbone_utils import build_semanticfpn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import numpy as np
import spacy
import fvcore.nn.weight_init as weight_init
from typing import Dict

from .my_Vis_Head import *
from .my_Fix_FPN import build_backbone
from .my_Fusion_Head import *
from .my_ASPP import *
from .my_Dynamic_Kernel_Head import *
from .my_Lang_Feat_Head import *



@META_ARCH_REGISTRY.register()
class my_Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device                 = torch.device(cfg.MODEL.DEVICE)
        # parameters
        self.cfg                    = cfg
        self.in_feature             = cfg.MODEL.FEATURE_ENCODER.IN_FEATURES
        # pre-process
        pixel_mean                  = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std                   = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer             = lambda x: (x - pixel_mean) / pixel_std
        # backbone
        self.backbone               = build_backbone(cfg)
        # Lang_head
        self.lang_feat_head         = Lang_Feat_Head(cfg)
        # Vis_head
        self.semantic_fpn           = build_semanticfpn(cfg, self.backbone.output_shape())
        self.feature_encoder        = FeatureEncoder(cfg)
        self.kernel_head            = KernelHead(cfg)
        # Ref_head
        self.fusion_head            = Fusion_Head(cfg)
        self.aspp_head              = ASPP_Head(cfg, 512)
        self.pred_head              = Softmax_Kernel_Head(cfg, 256, 64)
        self.out_sem                = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        # inference
        self.ret_thres              = cfg.MODEL.INFERENCE.RET_THRES
        self.to(self.device)


    def get_backbone_feature(self,batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.training==False:
            images[0]=self.resize_and_pad(images[0],416,416)
        images = [self.normalizer(x) for x in images]
        images = torch.stack(images)
        features,fpn_feats = self.backbone(images)
        return features,fpn_feats

    def get_vis_feature(self,features,fpn_feats):
        encode_feat = self.semantic_fpn(fpn_feats)
        encode_feat = self.feature_encoder(encode_feat)
        pred_weights = self.kernel_head(fpn_feats['p3'])
        return encode_feat,pred_weights

    def forward(self, batched_inputs):
        text_feats = self.lang_feat_head(batched_inputs) # get referring feature

        features,fpn_feats = self.get_backbone_feature(batched_inputs)
        encode_feat, pred_weights = self.get_vis_feature(features,fpn_feats)
        
        modal_feats = self.fusion_head(features,text_feats)
        modal_feats = self.aspp_head(modal_feats)
        
        stage_pred = self.out_sem(modal_feats)
        
        final_pred = self.pred_head([stage_pred],[pred_weights],encode_feat)

        ground_truth = [x["sem_seg"].to(self.device) for x in batched_inputs]
        ground_truth = torch.stack(ground_truth).unsqueeze(1)

        if self.training:
            H,W=ground_truth.shape[-2:]
            stage_pred = F.interpolate(stage_pred,size=(H,W),mode='bilinear', align_corners=False)
            final_pred = F.interpolate(final_pred,size=(H,W),mode='bilinear', align_corners=False)
            stage_loss = 0.5 * F.binary_cross_entropy_with_logits(stage_pred,ground_truth, reduction='mean')
            final_loss = F.binary_cross_entropy_with_logits(final_pred, ground_truth, reduction='mean')
            
            loss={}
            loss['stage_loss'] = stage_loss
            loss['final_loss'] = final_loss
            return loss
        else:
            pred_region=self.resize_and_crop(final_pred,ground_truth.shape[-2],ground_truth.shape[-1])
            pred_region=pred_region.sigmoid()
            pred_region[pred_region>self.ret_thres]=1
            pred_region[pred_region<=self.ret_thres]=0
            pred_region=pred_region.int()
            return [{"sem_seg":pred_region[0][0],"pred_regions":[stage_pred]}]
    
    @torch.no_grad()
    def resize_and_crop(self, im, input_h, input_w):
        im_h, im_w = im.shape[-2:]
        scale = max(input_h / im_h, input_w / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        crop_h = int(np.floor(resized_h - input_h) / 2)
        crop_w = int(np.floor(resized_w - input_w) / 2)
        resized_im=F.interpolate(im,size=[resized_h, resized_w], mode='bilinear', align_corners=False)        
        return resized_im[...,crop_h:crop_h+input_h, crop_w:crop_w+input_w]

    @torch.no_grad()
    def resize_and_pad(self, im, input_h, input_w):
        im_h, im_w = im.shape[-2:]
        scale = min(input_h / im_h, input_w / im_w)
        resized_h = int(np.round(im_h * scale))
        resized_w = int(np.round(im_w * scale))
        pad_h = int(np.floor(input_h - resized_h) / 2)
        pad_w = int(np.floor(input_w - resized_w) / 2)
        resized_im = F.interpolate(im.unsqueeze(0),size=[resized_h, resized_w], mode='bilinear', align_corners=False).squeeze(0)
        new_im=torch.zeros(3,input_h,input_w).cuda()
        new_im[...,pad_h:pad_h+resized_h, pad_w:pad_w+resized_w] = resized_im
        return new_im
