import os

import numpy as np
import pycocotools.mask as rletools
from pycocotools import coco
import cv2
from detectron2.structures import BoxMode


def load_sequences(path, img_path):
    objects_per_frame_per_sequence = {}
    cc = 0
    all_dictionaries = []
    for filename in os.listdir(path):
        cc += 1
        # print(cc)
        seq = filename.split('.')[0]
        seq_path_folder = os.path.join(path, seq)
        seq_path_txt = os.path.join(path,filename)
        if os.path.isdir(seq_path_folder):
            print('INPUT NOT A TXT!')
        elif os.path.exists(seq_path_txt):
            dataset_dict = load_txt(seq_path_txt, os.path.join(img_path, seq))
            all_dictionaries = np.concatenate((all_dictionaries,dataset_dict),axis=0)
        else:
            assert False, "Can't find data in directory " + path

    return objects_per_frame_per_sequence,all_dictionaries

def load_txt(path, img_path):
    useMask = False
    name = 'bbox_'+path.split('/')[-1]
    objects_per_frame = {}
    track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    allAnnots = []
    frame_prev = -1
    first = 0
    dataset_dicts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")
            # print(fields)
            frame = int(fields[0])

            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            # print(class_id)
            if not(class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}

            obj = []
            if frame != frame_prev:
                if first != 0:
                    record["annotations"] = objs
                    dataset_dicts.append(record)
                objs = []
                record = {}
                record["image_id"] = frame
                record["file_name"] = getImagePath(img_path, str(frame))
                record["height"] = mask['size'][0]
                record["width"] = mask['size'][1]

                if class_id != 10:
                    obj = {
                        "bbox": list(rletools.toBbox(mask)),
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": mask,
                        "category_id": class_id-1,
                        "iscrowd": 0
                    }
            else:
                if class_id != 10:
                    obj = {
                        "bbox": list(rletools.toBbox(mask)),
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": mask,
                        "category_id": class_id-1,
                        "iscrowd": 0
                    }

            if obj != []:
                objs.append(obj)
            frame_prev = frame
            first = 1

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def getImagePath(folder_path, s_numIm):
    seq = folder_path.split('/')[-1]
    
    if len(s_numIm) == 1:
        s_numIm = '000' + s_numIm
    elif len(s_numIm) == 2:
        s_numIm = '00' + s_numIm
    elif len(s_numIm) == 3:
        s_numIm = '0' + s_numIm
    if int(seq) >= 21:
        img = '00' + s_numIm + '.jpg'
    else:
        img = '00' + s_numIm + '.png'
    filename = os.path.join(folder_path, str(img))

    return filename


def get_KITTI_MOTS(img_path,gt_folder):
    print("Loading ground truth...")
    gt,all_dictionaries = load_sequences(gt_folder, img_path)
    return all_dictionaries
