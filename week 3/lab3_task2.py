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


DATASET_DIR_TEST = '/home/mcv/datasets/KITTI-MOTS/testing/image_02'
net_file = 'faster'  # choose between <retina> and <faster>


print('START')
# Create config with the pretrained model
cfg = get_cfg()

if net_file == 'faster':
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
elif net_file == 'retina':
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

# Get qualitative results
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
        if np.max(im[:,:,0])<=1:
            im = im*255
        if im is not None:
            outputs = predictor(im)
            print('OUT:',outputs["instances"].pred_classes)
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu")) # Passing the predictions to CPU from the GPU
            
        else:
            print('imgs are none')
        count += 1
        if count < 3:
            plt.imshow(v.get_image()[:, :, ::-1])
            plt.show()


