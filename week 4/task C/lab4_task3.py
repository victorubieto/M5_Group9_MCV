# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from dictionaries import get_KITTI_MOTS
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

from LossEvalHook import ValidationLoss

DATA_PATH_GT = '/home/group09/code/week4/task3/KITTI_MOTS'
DATASET_IMG_DIR = '/home/mcv/datasets/KITTI-MOTS/training/image_02'

## HYPERPARAMS
lr_sched = True
anchor_gen = False
iou_thr = False
experiment_type='_lr_s_500750_05'

# model_name = 'CityscapesDataset'
model_name = 'ResNet50-FPN'
trainData = 'KITTI-MOTS'# just used for the name of the output folder
datasets = ['KITTI-MOTS']

if model_name == 'ResNet50-FPN':
    config ="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
elif model_name == 'CityscapesDataset':
    config = "Cityscapes/mask_rcnn_R_50_FPN.yaml"


# Get pre-trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

# Data loaders
# kitti_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
thing_classes = ["Car", "Pedestrian"]
# print(thing_classes)
for data in datasets:
    for d in ['train', 'val']:
        DatasetCatalog.register(data+'_' + d, lambda d=d: get_KITTI_MOTS(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, d)))
        MetadataCatalog.get(data+'_'+d).set(thing_classes=["Car", "Pedestrian"])

# Setting training hyperparameters
cfg.DATASETS.TRAIN = ("KITTI-MOTS_train",)
cfg.DATASETS.VAL = ("KITTI-MOTS_val",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4

if not lr_sched:
    cfg.SOLVER.BASE_LR = 0.0025  # normal lr
if lr_sched:
    # cfg.SOLVER.LR_SCHEDULER = 0.0025
    # cfg.SOLVER.WARMUP_FACTOR = 1e-4
    cfg.SOLVER.STEPS = [0,    500, 750]
    cfg.SOLVER.GAMMA = 0.5

if anchor_gen:
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128, 256, 512,1024]] # [[16, 32, 64, 128, 265]]#[[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] # [[0.25, 0.5, 1.0]]
if iou_thr:
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.4, 0.7]


cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
cfg.TEST.EVAL_PERIOD = 500
cfg.INPUT.MASK_FORMAT = 'bitmask'


# Test lrshedule 1
# cfg.SOLVER.STEPS = [0,    500]
# cfg.SOLVER.GAMMA = 0.1

# Test lrshedule 2
# cfg.SOLVER.STEPS = [0,    500,    750]
# cfg.SOLVER.GAMMA = 0.1

# Test tresholds 1
# cfg.MODEL.RPN.IOU_THRESHOLDS = [0.4, 0.7]

# Test tresholds 2
# cfg.MODEL.RPN.IOU_THRESHOLDS = [0.4, 0.8]


cfg.OUTPUT_DIR = os.path.join(model_name + experiment_type)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])
# swap the order of PeriodicWriter and ValidationLoss
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
trainer.resume_or_load(resume=False)
trainer.train()


# Inference
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

evaluator = COCOEvaluator("KITTI-MOTS_val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR,"eval"))
val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))



# Get qualitative images
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
kitti_metadata = MetadataCatalog.get("KITTI-MOTS_val")

dataset_dicts = get_KITTI_MOTS(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, 'val'))
count = 0
random.seed(42)
for rand_d in random.sample(list(dataset_dicts), 100):
    img = cv2.imread(rand_d["file_name"])
    output = predictor(img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)

    instances = output["instances"].to("cpu")
    class_detections = instances[(instances.pred_classes == 0) | (instances.pred_classes == 2)]
    out = visualizer.draw_instance_predictions(class_detections)

    outpath = os.path.join(cfg.OUTPUT_DIR,"output_imgs")
    os.makedirs(outpath, exist_ok=True)
    cv2.imwrite(outpath + "/outputshow"+str(count)+".png", out.get_image()[:, :, ::-1])
    count += 1