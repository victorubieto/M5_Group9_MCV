# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
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
DATASET_IMG_DIR = '/home/mcv/datasets/out_of_context'

# LIST OF MODEL NAMES:
model_name = 'mask'

if model_name == 'mask':
    config ="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
elif model_name == 'faster':
    config ="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


print('START')
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(config)) 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
predictor = DefaultPredictor(cfg)
print('MODELWHEIGHTS',	cfg.MODEL.WEIGHTS)
count = 0
print('Try with Out_of context')


count = 0
for filename in os.listdir(DATASET_IMG_DIR):
    print('file',filename)
    im = plt.imread(os.path.join(DATASET_IMG_DIR,filename))
    if im is not None:
        outputs = predictor(im)
        print('OUT:',outputs["instances"].pred_classes)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu")) #Passing the predictions to CPU from the GPU
        
    else:
        print('imgs are none')
    count+=1
    plt.figure()
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig(filename)
