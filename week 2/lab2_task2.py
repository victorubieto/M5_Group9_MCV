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


DATASET_DIR_TRAIN = '/home/group09/code/Dataset_withVal/train'
DATASET_DIR_TEST = '/home/group09/code/week1/Dataset_withVal/test/'


print('START')
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
print('MODELWHEIGHTS',	cfg.MODEL.WEIGHTS)
count = 0
print('Try with MIT_SPLIT')
for classfolder in os.listdir(DATASET_DIR_TEST):
    print(classfolder)
    newpath = os.path.join(DATASET_DIR_TEST,classfolder)
    print('newpath',newpath)
    count = 0
    for filename in os.listdir(newpath):
        print('file',filename)
        im = plt.imread(os.path.join(newpath,filename))
        if im is not None:
            outputs = predictor(im)
            print('OUT:',outputs["instances"].pred_classes)
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu")) #Passing the predictions to CPU from the GPU
            
        else:
            print('imgs are none')
        count+=1
        if count<3:
            plt.imshow(v.get_image()[:, :, ::-1])
            plt.show()


