# Object Detection, Recognition and Segmentation
 
This is the repository of the Group 9 composed by Mar Ferrer (mar.ferrerf@e-campus.uab.cat), Alex Tempelaar (alexander.tempelaar@e-campus.uab.cat), and Víctor Ubieto (victor.ubieto@e-campus.uab.cat).

In this project we provide a nice implementation of a Image Classification network using the framework from Pytorch.

You can find the final documentation and report of the whole project [here](https://).

Moreover, you can check the weekly slides in the folowing links:
* [Week 1](https://docs.google.com/presentation/d/1deTzukazFwTeO4joZ5Fpa5-4wPc_g2MzVJk3duT0UfQ/edit?usp=sharing)
* [Week 2](https://docs.google.com/presentation/d/1SV-Jc5rKc0fMkfuLtEqk-vGWJdk6FSVr9HdRkDZe5k4/edit?usp=sharing)
* Week 3
* Week 4
* Week 5

# Code Explanation

## Week 1

This code is a Pytorch implementation of an image classiffier based on CNNs. The used model is extracted from a previous module (M3) in which we worked with Keras. So the aim of this project is to spot the differences between the two frameworks. 
In order to run this code, you have to do the following:

1. Open the main file, lab1.py, and insert your path for the training set: DATASET_DIR_TRAIN: --Line 18
2. Insert your path for the training set: DATASET_DIR_TEST: --Line 19
3. Run: >> python lab1.py

## Week 2

This code is a first approach to Detectron2 framework where we used 2 differennt pretrained models to detect objects in our datasets. Additionally, we finetuned a Faster R-CNN model using the KITTI dataset. Finally, the week folder also includes a brief report about Object Detection, Recognition and Segmentation.

For tasks 1-3:
1. Open the main file of the desired task, lab2_taskX.py, and insert your path for the training set: DATASET_DIR_TRAIN: --Line 22
2. Insert your path for the training set: DATASET_DIR_TEST: --Line 23
3. Run: >> python lab2_taskX.py

For task 4:
1. Open the file lab2_task4.py, and insert your path to the folder with the .txt files representing the three splits of the data: DATASET_ROOT_DIR: --Line 27
2. Insert your path to the folder where are stored all the images: DATASET_IMG_DIR: --Line 28
3. Insert your path to the folder where are stored the .txt with the detections of each image: DATASET_DET_DIR: --Line 29
4. Run: >> python lab2_task4.py
