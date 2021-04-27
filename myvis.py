from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from panopticfcn import add_panopticfcn_config, build_lr_scheduler
os.environ["NCCL_LL_THRESHOLD"] = "0"
from detectron2.data import MetadataCatalog
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from Ref_datasets import get_ref_dataset,Ref_mapper
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from Ref_evaluator import Ref_SemSegEvaluator
from Ref_dataset_mapper import Ref_DatasetMapper
from train import Register_datasets
from detectron2.engine.defaults import DefaultPredictor
import random
import cv2
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2.checkpoint import DetectionCheckpointer
import torchvision
import torch
from detectron2.data.detection_utils import read_image
from detectron2.data import detection_utils as utils

# usage : python myvis.py --config-file configs/PanopticFCN-R50-1x-FAST-ref.yaml --num-gpus 1 


def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:]), "{},{}".format(target.shape, masks.shape)
    temp = (masks * target)
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection.item(),union.item()


if __name__=="__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = get_cfg()
    add_panopticfcn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.RUNNING_MODE="vis"

    cfg.freeze()
    default_setup(cfg, args)

    Register_datasets(cfg.DATASETS.TRAIN)
    if cfg.DATASETS.TEST[0]!=cfg.DATASETS.TRAIN[0]:
        Register_datasets(cfg.DATASETS.TEST)

    # predictor = DefaultPredictor(cfg)
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    testset=cfg.DATASETS.TEST[0]
    dataset_dicts = build_detection_test_loader(cfg, testset,mapper=Ref_DatasetMapper(cfg, False))
    
    sum_iou=0
    sum_ii=0
    sum_oo=0
    num=0
    for batch_data in dataset_dicts:  
        d=batch_data[0]

        outputs = model([d])
        pred = outputs[0]["sem_seg"].cpu()
        gt = d["sem_seg"]
        ii,oo=compute_mask_IU(pred, gt)
        iou=ii/oo
        sum_iou+=iou
        num+=1
        print("idx = {}, cnt_iou = {}, mean_iou = {}".format(num,iou,sum_iou/num))
          
        for i in range(len(outputs[0]["pred_regions"])):
            torchvision.utils.save_image(outputs[0]["pred_regions"][i][0],'pred{}.jpg'.format(i))    
    
        image=d["image"]
        image=image.permute(1,2,0)
        image=utils.convert_image_to_rgb(image, "BGR")
       
        v = Visualizer(image,metadata=MetadataCatalog.get(testset))
        v.draw_sem_seg(gt.int().to("cpu"))
        v.get_output().save('fig-gt.jpg')


        v = Visualizer(image,metadata=MetadataCatalog.get(testset))
        v.draw_sem_seg(outputs[0]["sem_seg"].to("cpu"))
        v.get_output().save('fig-output.jpg')

        v = Visualizer(image,metadata=MetadataCatalog.get(testset))
        v.get_output().save('fig-o.jpg')

        print(d["phrase"])
        input()