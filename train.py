import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import json
from detectron2.structures import BoxMode

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
print(torch.__version__, torch.cuda.is_available())
setup_logger()
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "./datasets/coco/annotations/instances_train2017.json", "./datasets/coco/train2017/")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

# SOLVER
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.LR_POLICY = "steps_with_decay"
cfg.SOLVER.STEPS = [5000, 7000, 9000]
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.WARMUP_FACTOR = 0.01
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.GAMMA = 0.1

# Train
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1, 2, 5]]
cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
cfg.TEST.DETECTIONS_PER_IMAGE = 500

cfg.INPUT.CROP.SIZE = [1.0, 1.0]
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
cfg.OUTPUT_DIR = './detectron2_results/'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
