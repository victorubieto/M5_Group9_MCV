# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from dictionaries import get_KITTI_dicts
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2 
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


DATASET_IMG_DIR  = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
DATA_PATH_GT = '/home/group09/code/week3/task3/instancesNew'


# Get pre-trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

#data loaders
kitti_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
thing_classes = kitti_metadata.thing_classes

for d in ['train', 'val']:
    DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: get_KITTI_dicts(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, d)))
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=thing_classes)

# Setting training hyperparameters
cfg.DATASETS.TRAIN = ("KITTI-MOTS_train",)
cfg.DATASETS.TEST = ("KITTI-MOTS_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.025  # pick a good LR
cfg.SOLVER.MAX_ITER = 100
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #(default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)

# Evaluate performance of pre-trained model
evaluator = COCOEvaluator("KITTI-MOTS_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))


