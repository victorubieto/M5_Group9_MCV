from detectron2.structures import BoxMode
import cv2
import os
from enum import Enum


def get_KITTI_dicts(img_dir, data_paths):

    dataset_dicts = []
    for text in os.listdir(data_paths):
        imP = text.split('_')[-1]
        imP = imP.split('.')[0]
        folder_path = os.path.join(img_dir,imP)
        numIm = 0
        s_numIm = '0'
        for k in range(len(os.listdir(folder_path))):
            if len(s_numIm) == 1:
                s_numIm = '000' + s_numIm
            elif len(s_numIm) == 2:
                s_numIm = '00' + s_numIm
            elif len(s_numIm) == 3:
                s_numIm = '0' + s_numIm

            img = '00' + s_numIm + '.png'
            record = {}
            filename = os.path.join(folder_path, str(img))
            # print(filename)
            height, width = cv2.imread(filename).shape[:2]
            record["image_id"] = int(img.split('.')[0])
            record["file_name"] = filename
            record["height"] = height
            record["width"] = width
            numIm += 1
            s_numIm = str(numIm)
            annots_path = os.path.join(data_paths,text)
            with open(annots_path) as f:
                lines = f.readlines()

            if len(lines) > k:
                dataL = lines[k].split('|')
                objs = []
                for j in range(len(dataL)):
                    box = dataL[j]
                    boxL = box.split(' ')
                    if j == 0:
                        if boxL[1] != '10':
                            IDval = 0
                            if boxL[1] == '1':
                                IDval = 2
                            elif boxL[1] == '2':
                                IDval = 0
                            obj = {
                                "bbox": [float(boxL[4]), float(boxL[5]), float(boxL[6]), float(boxL[7])],
                                "bbox_mode": BoxMode.XYWH_ABS,
                                "category_id": IDval,
                                "iscrowd": 0
                            }
                    else:
                        if boxL[0] != '10':
                            IDval = 0
                            if boxL[0] == '1':
                                IDval = 2
                            elif boxL[0] == '2':
                                IDval = 0
                            obj = {
                                "bbox": [float(boxL[3]), float(boxL[4]), float(boxL[5]), float(boxL[6])],
                                "bbox_mode": BoxMode.XYWH_ABS,
                                "category_id": IDval,
                                "iscrowd": 0
                            }
            
                    objs.append(obj)
            
                record["annotations"] = objs
                dataset_dicts.append(record)
                
    return dataset_dicts

