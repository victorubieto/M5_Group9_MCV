# Object Detection, Recognition and Segmentation
 
This is the repository of the Group 9 composed by Mar Ferrer (mar.ferrerf@e-campus.uab.cat), Alex Tempelaar (alexander.tempelaar@e-campus.uab.cat), and Víctor Ubieto (victor.ubieto@e-campus.uab.cat).

In this project we provide a nice implementation of a Image Classification network using the framework from Pytorch.

You can find the documentation and report of the project [here](https://www.overleaf.com/read/xfbwshstznzh).

Moreover, you can check the weekly slides in the folowing links:
* [Week 1](https://docs.google.com/presentation/d/1deTzukazFwTeO4joZ5Fpa5-4wPc_g2MzVJk3duT0UfQ/edit?usp=sharing)
* [Week 2](https://docs.google.com/presentation/d/1SV-Jc5rKc0fMkfuLtEqk-vGWJdk6FSVr9HdRkDZe5k4/edit?usp=sharing)
* [Week 3](https://docs.google.com/presentation/d/17VsHTOEuCXAYGFqN3LIry-6TbTCovvluaCs0u-U4e7U/edit?usp=sharing)
* [Week 4](https://docs.google.com/presentation/d/1S3ea5e--dl1P9OR8gdf2czNSOtJtaC24RTz5bqNrZOk/edit?usp=sharing)
* Week 5

# Code Explanation

## Week 1

This code is a Pytorch implementation of an image classiffier based on CNNs. The used model is extracted from a previous module (M3) in which we worked with Keras. So the aim of this project is to spot the differences between the two frameworks. 
In order to run this code, you have to do the following:

1. Open the main file, lab1.py, and insert your path for the training set: DATASET_DIR_TRAIN: --Line 18
2. Insert your path for the test set: DATASET_DIR_TEST: --Line 19
3. Run: >> python lab1.py

## Week 2

This code is a first approach to Detectron2 framework where we used 2 differennt pretrained models to detect objects in our datasets. Additionally, we finetuned a Faster R-CNN model using the KITTI dataset. Finally, the week folder also includes a brief report about Object Detection, Recognition and Segmentation.

For tasks 1-3:
1. Open the main file of the desired task, lab2_taskX.py, and insert your path for the training set: DATASET_DIR_TRAIN: --Line 22
2. Insert your path for the test set: DATASET_DIR_TEST: --Line 23
3. Run: >> python lab2_taskX.py

For task 4:
1. Open the file lab2_task4.py, and insert your path to the folder with the .txt files representing the three splits of the data: DATASET_ROOT_DIR: --Line 27
2. Insert your path to the folder where are stored all the images: DATASET_IMG_DIR: --Line 28
3. Insert your path to the folder where are stored the .txt with the detections of each image: DATASET_DET_DIR: --Line 29
4. Run: >> python lab2_task4.py

## Week 3

For task 2:
1. Open the file lab3_task2.py, and insert your path for the test set: DATASET_DIR_TEST: --Line 22
2. Choose between Faster R-CNN and RetinaNet models by writting 'faster' or 'retina': --Line 23
3. Run: >> python lab3_task2.py

For task 3:
1. Open the file lab3_task3.py, and insert your path to the folder with images of your dataset: DATASET_IMG_DIR: --Line 27
2. Insert your path to the folder where are stored all the annotations (ground truth) divided into two folders, one for training and the other for validation: DATA_PATH_GT: --Line 28
4. Run: >> python lab3_task3.py 

For task 4 and 5:
1. Open the file lab3_task4.py, and insert your path to the folder with images of your dataset: DATASET_IMG_DIR: --Line 53
2. Insert your path to the folder where are stored all the annotations (ground truth) divided into three folders, one for training, one for validation, and one for test: DATA_PATH_GT: --Line 54
3. Choose the desired model and dataset between "faster" and "retina", and between "MOTS" and "KITTI": --Line 55, 56
4. Run: >> python lab3_task4.py 

## Week 4

For task A:
1. Open the file lab4_task1.py, and insert your path for the test set: Groundtruth dataset: --Line 28
2. Put dataset for the image path: --Line 29
3. Chose a model name (uncomment one): --Lines 32-39
5. Run: >> python lab4_task1.py

For task B:
1. Open the file lab4_task1.py, and insert your path for the test set: Groundtruth dataset: --Line 42
2. Put dataset for the image path: --Line 43
3. Chose a model name (uncomment one): --Lines 31
4. Output folder name: --Line 33
5. Run: >> python lab4_task2.py

For task C:
1. Open the file lab4_task1.py, and insert your path for the test set: Groundtruth dataset: --Line 28
2. Put dataset for the image path: --Line 29
3. Set which hyperparameters you want to use: --Lines 32-34
4. Output folder name (experiment name): --Line 35
5. Model name: --Line 38
6. Run: >> python lab4_task3.py
