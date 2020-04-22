# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:42:43 2020

@author: astri
"""

import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split

# from os import path
# current path is fetched
current_path = os.getcwd()
# folder is fetched where images are located dynamically
folder = os.path.join(current_path,
                             'VOCtrainval_11-May-2009',
                             'VOCdevkit',
                             'VOC2009')
# Read list of all files for classification
class_file = open(os.path.join(folder, 'ImageSets', 'Main', 'trainval.txt'))
class_list = class_file.read().splitlines()

# Read list of all files for segmentation
segm_file = open(os.path.join(folder, 'ImageSets', 'Segmentation', 'trainval.txt'))
segm_list = segm_file.read().splitlines()
# Split data for training, validation and testing
segm_train_list, segm_valtest_list = train_test_split(segm_list, test_size=0.5, random_state=42)
segm_val_list, segm_test_list = train_test_split(segm_valtest_list, test_size=0.5, random_state=42)

with open('segm_train.txt', 'w') as f:
    for item in segm_train_list:
        f.write("%s\n" % item)
with open('segm_val.txt', 'w') as f:
    for item in segm_val_list:
        f.write("%s\n" % item)
with open('segm_test.txt', 'w') as f:
    for item in segm_test_list:
        f.write("%s\n" % item)


