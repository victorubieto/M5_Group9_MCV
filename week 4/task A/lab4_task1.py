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

DATA_PATH_GT = '/home/group09/code/week4/task1_alex/KITTI_MOTS'
DATASET_IMG_DIR = '/home/mcv/datasets/KITTI-MOTS/training/image_02'

# LIST OF MODEL NAMES:
# model_name = 'ResNet50-FPN'
# model_name = 'ResNet101-FPN'
# model_name = 'ResNeXt101-FPN'
# model_name = 'R50-C4'
# model_name = 'R101-C4'
# model_name = 'R50-DC5'
model_name = 'R101-DC5'
# model_name = 'CityscapesDataset'

if model_name == 'ResNet50-FPN':
    config ="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
elif model_name == 'ResNet101-FPN':
    config ="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
elif model_name == 'ResNeXt101-FPN':
    config = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
elif model_name == 'R50-C4':
    config = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
elif model_name == 'R101-C4':
    config = "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
elif model_name == 'R50-DC5':
    config = "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
elif model_name == 'R101-DC5':
    config = "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
elif model_name == 'CityscapesDataset':
    config = "Cityscapes/mask_rcnn_R_50_FPN.yaml"

# Get pre-trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config) 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

# Data loaders
kitti_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
thing_classes = kitti_metadata.thing_classes


for d in ['train', 'val']:
    DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: get_KITTI_MOTS(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, d)))
    MetadataCatalog.get('KITTI-MOTS_'+d).set(thing_classes=thing_classes)

# Setting training hyperparameters
cfg.DATASETS.TRAIN = ("KITTI-MOTS_train",)
cfg.DATASETS.VAL = ("KITTI-MOTS_val",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.025  # pick a good LR
cfg.SOLVER.MAX_ITER = 100
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   #(default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
cfg.TEST.EVAL_PERIOD = 500
cfg.INPUT.MASK_FORMAT = 'bitmask'


cfg.OUTPUT_DIR = os.path.join(model_name)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])
# swap the order of PeriodicWriter and ValidationLoss
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
trainer.resume_or_load(resume=False)
# trainer.train()


# Inference
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

evaluator = COCOEvaluator("KITTI-MOTS_val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR,"eval"))
val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))



#Get qualitative images
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
kitti_metadata = MetadataCatalog.get("KITTI-MOTS_val")

dataset_dicts = get_KITTI_MOTS(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, 'val'))
count = 0
random.seed(42)
for rand_d in random.sample(list(dataset_dicts), 500):
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