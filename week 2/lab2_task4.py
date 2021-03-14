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


DATASET_ROOT_DIR = '/home/mcv/datasets/KITTI'
DATASET_IMG_DIR = '/home/mcv/datasets/KITTI/data_object_image_2/training/image_2'
DATASET_DET_DIR = '/home/mcv/datasets/KITTI/training/label_2'


print('START')
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
print('MODELWHEIGHTS',	cfg.MODEL.WEIGHTS)

# Add metadata to
for d in ["train", "val"]:
    DatasetCatalog.register("KITTI_" + d, lambda d=d: get_KITTI_dicts(DATASET_IMG_DIR, DATASET_DET_DIR, os.path.join(DATASET_ROOT_DIR, d + '_kitti.txt')))
    MetadataCatalog.get("KITTI_" + d).set(thing_classes=["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"])
kitti_metadata = MetadataCatalog.get("KITTI_train")


dataset_dicts = get_KITTI_dicts(DATASET_IMG_DIR, DATASET_DET_DIR, os.path.join(DATASET_ROOT_DIR,'train_kitti.txt'))
for rand_d in random.sample(dataset_dicts, 1):
    img = cv2.imread(rand_d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)
    print(rand_d)
    out = visualizer.draw_dataset_dict(rand_d)
    # cv2.imshow(out.get_image()[:, :, ::-1])
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
    cv2.imwrite("outputshow.png", out.get_image()[:, :, ::-1])

#------------------------TRAIN-----------------------------------------
print('TRAIN')
from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("KITTI_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   #  (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

#------------------------------------INFERENCE--------------------------------
print('INFERENCE')
evaluator = COCOEvaluator("KITTI_val", cfg, False, output_dir="./output_alex/")
print('ha passat base 1')
val_loader = build_detection_test_loader(cfg, "KITTI_val")
print('ha passat base 2')
print(inference_on_dataset(trainer.model, val_loader, evaluator))

