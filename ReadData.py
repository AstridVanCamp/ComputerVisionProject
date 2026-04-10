# -*- coding: utf-8 -*-
"""
Functions for reading images from a given filelist and
gathering these in an array
"""

import cv2
import os
import numpy as np

def get_JPEGimages(filename, image_dir, resize=False):
    # Function for creating an array of the images
    image_list = open(filename)
    images = []
    for line in image_list:
        file = os.path.join(image_dir, 'JPEGImages', line.strip()+'.jpg')
        img = cv2.imread(file)
        
        if resize:
            nrows = 256
            ncols = 256
            img = cv2.resize(img, (ncols,nrows))
            
        images.append(img/255)
        
    return np.asarray(images)


def get_PNGsegments(filename, image_dir, resize=False):
    # Function for creating an array of the segmentation masks
    image_list = open(filename)
    images = []
    for line in image_list:
        file = os.path.join(image_dir, 'SegmentationClass', line.strip()+'.png')
        img = cv2.imread(file)
        
        if resize:
            nrows = 256
            ncols = 256
            img = cv2.resize(img, (ncols,nrows), interpolation=cv2.INTER_NEAREST)

        images.append(img)   
        
    return np.asarray(images)

