# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:42:43 2020

@author: astri
"""

import cv2
import os

def get_JPEGimages(filename, image_dir):
    image_list = open(filename)
    images = list()
    for line in image_list:
        file = os.path.join(image_dir, 'JPEGImages', line.strip()+'.jpg')
        img = cv2.imread(file)
        images.append(img)    
    
    return images

def get_PNGsegments(filename, image_dir):
    image_list = open(filename)
    images = list()
    for line in image_list:
        file = os.path.join(image_dir, 'SegmentationClass', line.strip()+'.png')
        img = cv2.imread(file)
        images.append(img)    
    
    return images
    

# current path is fetched
current_path = os.getcwd()
# folder is fetched where images are located dynamically
folder = os.path.join(current_path,
                             'VOCtrainval_11-May-2009',
                             'VOCdevkit',
                             'VOC2009')

class_test_images = get_JPEGimages('class_test.txt', folder)

segm_test_objects = get_PNGsegments('segm_test.txt', folder)
