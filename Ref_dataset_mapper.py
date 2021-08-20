import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["Ref_DatasetMapper"]


class Ref_DatasetMapper:

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        # fmt: off
        self.is_train               = is_train
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = {
            "is_train": is_train,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        npz_filemap = np.load(dataset_dict)
        batch = dict(npz_filemap)
        npz_filemap.close()
        
        dataset_dict={}
        
        image=batch['im_batch'].astype(np.float32)
        image=image[:,:,::-1]
        image=torch.Tensor(image.copy())
        image=torch.as_tensor(np.ascontiguousarray(image.permute(2, 0, 1)))
        dataset_dict["image"]=image
    
        mask=batch['mask_batch'].astype(np.float32)
        mask=torch.Tensor(mask)
        dataset_dict["sem_seg"]=mask
        dataset_dict["ori_seg"]=mask

        dataset_dict["phrase"]=batch['sent_batch'][0]
        if isinstance(dataset_dict["phrase"],np.str_):
            dataset_dict["phrase"] = str(dataset_dict["phrase"])
        else:
            dataset_dict["phrase"] = dataset_dict["phrase"].decode('utf-8')

        dataset_dict["text"]=torch.Tensor(batch['text_batch'])

        dataset_dict['width']=mask.shape[0]
        dataset_dict['height']=mask.shape[1]
  
        return dataset_dict
