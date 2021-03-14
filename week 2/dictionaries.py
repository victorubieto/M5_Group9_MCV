from detectron2.structures import BoxMode
import cv2
import os
from enum import Enum


def label2int(label):
    if label == 'Car':
        id = 0
    elif label == 'Van':
        id = 1
    elif label == 'Truck':
        id = 2
    elif label == 'Pedestrian':
        id = 3
    elif label == 'Person_sitting':
        id = 4
    elif label == 'Cyclist':
        id = 5
    elif label == 'Tram':
        id = 6
    elif label == 'Misc':
        id = 7
    else:
        id = 8

    return id


def get_KITTI_dicts(img_dir, det_dir, data_paths):

    with open(data_paths) as fimg:
        images = fimg.readlines()

    dataset_dicts = []
    for idx in range(len(images)):
        record = {}

        filename = os.path.join(img_dir, images[idx][:-5] + '.png')
        height, width = cv2.imread(filename).shape[:2]


        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        with open(os.path.join(det_dir,images[idx][:-1])) as fdet:
            detections = fdet.readlines()

        objs = []
        for det in range(len(detections)):
            spt_line = detections[det].split(' ')
            obj = {
                "bbox": [float(spt_line[4]), float(spt_line[5]), float(spt_line[6]), float(spt_line[7])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": label2int(spt_line[0])
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
