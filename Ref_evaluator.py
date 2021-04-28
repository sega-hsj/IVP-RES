import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator


class Ref_SemSegEvaluator(DatasetEvaluator):

    def __init__(
        self, dataset_name, distributed, output_dir=None, *, num_classes=None, ignore_label=None
    ):
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        meta = MetadataCatalog.get(dataset_name)
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []
        self.sum_iou=0
        self.num=0
        self.prec=np.zeros(5,dtype=np.int64)
        self.thres=np.array([0.5,0.6,0.7,0.8,0.9])

    def compute_mask_IU(self,masks, target):
        assert(target.shape[-2:] == masks.shape[-2:])
        temp = (masks * target)
        intersection = temp.sum()
        union = ((masks + target) - temp).sum()
        return intersection.item(),union.item()

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            gt=input["sem_seg"].to(self._cpu_device)
            gt=np.array(gt,dtype=np.int)
            output=output["sem_seg"].to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            gt[gt == self._ignore_label] = self._num_classes

            ci,cu=self.compute_mask_IU(pred,gt)
            iou=ci/cu
            self.sum_iou+=iou
            self.num+=1
            self.prec[self.thres<(iou)]+=1


    def evaluate(self):
        if self._distributed:
            synchronize()
         
            self._sum_iou=all_gather(self.sum_iou)
            self._num=all_gather(self.num)
            self._prec=all_gather(self.prec)
            if not is_main_process():
                return

            self.sum_iou=0
            self.num=0
            self.prec=np.zeros(5,dtype=np.int64)

            for i in self._prec:
                self.prec+=i

            for c,d in zip(self._sum_iou,self._num):
                self.sum_iou+=c
                self.num+=d
        
        res={}
        res["IoU"]=self.sum_iou/self.num*100
        res["prec0.5"]=self.prec[0]/self.num*100
        res["prec0.6"]=self.prec[1]/self.num*100
        res["prec0.7"]=self.prec[2]/self.num*100
        res["prec0.8"]=self.prec[3]/self.num*100
        res["prec0.9"]=self.prec[4]/self.num*100
        result = OrderedDict({"ref_seg": res})
        self._logger.info(result)
        return result 
        
   
