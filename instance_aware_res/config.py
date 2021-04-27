# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_instance_aware_res_config(cfg):

    cfg.MODEL.TENSOR_DIM                 = 20
    cfg.MODEL.IGNORE_VALUE               = 255
    cfg.SOLVER.POLY_LR_POWER             = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING   = 0.0

    cfg.MODEL.SEMANTIC_FPN   = CN()
    cfg.MODEL.SEMANTIC_FPN.IN_FEATURES   = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.SEMANTIC_FPN.CONVS_DIM     = 256
    cfg.MODEL.SEMANTIC_FPN.COMMON_STRIDE = 4
    cfg.MODEL.SEMANTIC_FPN.NORM          = "GN"

    cfg.MODEL.KERNEL_HEAD    = CN()
    cfg.MODEL.KERNEL_HEAD.NUM_CONVS       = 3
    cfg.MODEL.KERNEL_HEAD.DEFORM          = False
    cfg.MODEL.KERNEL_HEAD.COORD           = True
    cfg.MODEL.KERNEL_HEAD.CONVS_DIM       = 256
    cfg.MODEL.KERNEL_HEAD.NORM            = "GN"

    cfg.MODEL.POSITION_HEAD   = CN()
    cfg.MODEL.POSITION_HEAD.NORM          = "GN"

    cfg.MODEL.FEATURE_ENCODER    = CN()
    cfg.MODEL.FEATURE_ENCODER.IN_FEATURES     = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FEATURE_ENCODER.NUM_CONVS       = 3
    cfg.MODEL.FEATURE_ENCODER.CONVS_DIM       = 64
    cfg.MODEL.FEATURE_ENCODER.DEFORM          = False
    cfg.MODEL.FEATURE_ENCODER.COORD           = True
    cfg.MODEL.FEATURE_ENCODER.NORM            = ""

    cfg.MODEL.INFERENCE      = CN()
    cfg.MODEL.INFERENCE.RET_THRES         = 0.2
    
    cfg.EMB_PATH=None
    cfg.USE_LANG_ATT=False
    cfg.DATASETS.ROOT=""