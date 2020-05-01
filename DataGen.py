# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:41:12 2020

@author: astri
"""

import cv2
import os
import numpy as np

from ReadData import get_JPEGimages, get_PNGsegments
from VOClabelcolormap import one_hot_encode, unique_class

from keras.utils import to_categorical

def data_gen(filelist, batch_size, num_classes):
    # current path is fetched
    current_path = os.getcwd()
    # folder is fetched where images are located dynamically
    folder = os.path.join(current_path,
                                  'VOCtrainval_11-May-2009',
                                  'VOCdevkit',
                                  'VOC2009')
    segm_resize = True
    segm_img = get_JPEGimages(filelist, folder, resize=segm_resize)
    segm_mask = get_PNGsegments(filelist, folder, resize=segm_resize).astype(np.uint8)
    
    c = 0  
    while (True):
        img = np.zeros((batch_size, 256, 256, 3)).astype('float')
        mask = np.zeros((batch_size, 256*256,22)).astype('float')
    
        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
            if i < len(segm_img):                    
                img[i-c] = segm_img[i] #add to array - img[0], img[1], and so on.            
                img_mask = unique_class(segm_mask[i], num_classes)
                mask[i-c] = to_categorical(img_mask, num_classes=22)

        c+=batch_size
    
        yield img, mask



train_gen = data_gen('segm_train.txt', 16, 20)
val_gen = data_gen('segm_val.txt', 16, 20)
test_gen = data_gen('segm_test.txt', 16, 20)