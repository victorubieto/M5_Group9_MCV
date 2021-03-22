# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from dictionariesT4 import get_KITTI_dicts, get_MOTS_dicts
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2 
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.utils.visualizer import ColorMode

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader

import torch
import detectron2.utils.comm as comm


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)


DATASET_IMG_DIR  = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
DATA_PATH_GT = '/home/group09/code/week3/task4_A/instancesNew'
net_model = 'faster' # choose between <faster> and <retina>
dataset = 'MOTS' # choose between <MOTS> and <KITTI>


net_file = ''
if net_model == 'faster':
    net_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
elif net_model == 'retina':
    net_file = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
else:
    print("The input of net_model must be <faster> or <retina>")


if dataset == 'MOTS':
    # This dataset only has the class Pedestrian
   
    # Add metadata to
    for d in ["train", "val"]:
        DatasetCatalog.register("MOTSChallenge_" + d, lambda d=d: get_MOTS_dicts(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, d)))
        MetadataCatalog.get("MOTSChallenge_" + d).set(thing_classes=["Pedestrian"])

    # Create/Load config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(net_file))
    cfg.DATASETS.TRAIN = ("MOTSChallenge_train",)
    cfg.DATASETS.VAL = ("MOTSChallenge_val",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(net_file)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 200
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # fast and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.OUTPUT_DIR = os.path.join("motschallenge_4_0025_200_128")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    # Train
    trainer.train()

    # Inference (quantitative results)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    if net_model == 'faster':
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set the testing threshold for this model
    elif net_model == 'retina':
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    
    evaluator = COCOEvaluator("MOTSChallenge_val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR,"eval"))
    val_loader = build_detection_test_loader(cfg, "MOTSChallenge_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    
    # Get qualitative results
    if net_model == 'faster':
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set the testing threshold for this model
    elif net_model == 'retina':
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    predictor = DefaultPredictor(cfg)
    kitti_metadata = MetadataCatalog.get("MOTSChallenge_val")
    
    dataset_dicts = get_MOTS_dicts(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, 'val'))
    count = 0
    for rand_d in random.sample(dataset_dicts, 30):
        img = cv2.imread(rand_d["file_name"])
        output = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)
        print(rand_d)
        out = visualizer.draw_instance_predictions(output["instances"].to("cpu"))
        
        os.makedirs("images", exist_ok=True)
        cv2.imwrite("images/outputshow" + str(count) + ".png", out.get_image()[:, :, ::-1])
        count += 1

elif dataset == 'KITTI':
    # In this case the dataset has two classes: Car and Pedestrian
    # It also performs the evaluation with the test sub-dataset
    
    # Add metadata to
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("KITTI-MOTS_" + d, lambda d=d: get_KITTI_dicts(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, d)))
        MetadataCatalog.get("KITTI-MOTS_" + d).set(thing_classes=["Car","Pedestrian"])

    # Just load the pretrained model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(net_file))
    cfg.DATASETS.TRAIN = ("KITTI-MOTS_train",)
    cfg.DATASETS.VAL = ("KITTI-MOTS_val",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(net_file) # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # fast and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    
    cfg.OUTPUT_DIR = os.path.join("kittimots_16_0025_500_128_testdata")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    # Train
    trainer.train()

    # Inference
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    if net_model == 'faster':
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set the testing threshold for this model
    elif net_model == 'retina':
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    
    evaluator = COCOEvaluator("KITTI-MOTS_test", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR,"eval"))
    val_loader = build_detection_test_loader(cfg, "KITTI-MOTS_test")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    
    # Get qualitative images
    if net_model == 'faster':
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set the testing threshold for this model
    elif net_model == 'retina':
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    kitti_metadata = MetadataCatalog.get("KITTI-MOTS_test")
    
    dataset_dicts = get_KITTI_dicts(DATASET_IMG_DIR,os.path.join(DATA_PATH_GT, 'test'))
    count = 0
    random.seed(42)
    for rand_d in random.sample(dataset_dicts, 30):
        img = cv2.imread(rand_d["file_name"])
        output = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_metadata, scale=0.5)
        out = visualizer.draw_instance_predictions(output["instances"].to("cpu"))
        
        os.makedirs("images", exist_ok=True)
        cv2.imwrite("images/outputshow" + str(count) + ".png", out.get_image()[:, :, ::-1])
        count += 1

else:
    print("The input of dataset must be <MOTS> or <KITTI>")

