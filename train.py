import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from instance_aware_res import add_instance_aware_res_config, build_lr_scheduler
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
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from Ref_datasets import get_ref_dataset
from Ref_evaluator import Ref_SemSegEvaluator
from Ref_dataset_mapper import Ref_DatasetMapper

def Register_datasets(name,root):
    name=name[0]
    dataset,split=name.split('_')
    DatasetCatalog.register(name,lambda: get_ref_dataset(\
    data_root=root,\
    dataset=dataset,\
    split=split))
    MetadataCatalog.get(name).stuff_classes=[i for i in range(2)]
    MetadataCatalog.get(name).evaluator_type='ref'
    MetadataCatalog.get(name).ignore_label=None
    print("finish dataset register of {}".format(name))


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls,cfg):
        return build_detection_train_loader(cfg,mapper=Ref_DatasetMapper(cfg, True))

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        # return build_detection_test_loader(cfg, dataset_name,mapper=Ref_mapper(cfg).mapping)
        return build_detection_test_loader(cfg, dataset_name,mapper=Ref_DatasetMapper(cfg, False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "ref":
            evaluator_list.append(
                Ref_SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=2,
                    ignore_label=None,
                    output_dir=output_folder,
                )
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

def setup(args):
    cfg = get_cfg()
    add_instance_aware_res_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    Register_datasets(cfg.DATASETS.TRAIN, cfg.DATASETS.ROOT)
    if cfg.DATASETS.TEST[0]!=cfg.DATASETS.TRAIN[0]:
       Register_datasets(cfg.DATASETS.TEST, cfg.DATASETS.ROOT)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(10000)
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )