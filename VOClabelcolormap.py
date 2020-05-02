# -*- coding: utf-8 -*-
""" 
Python implementation of the color map function for the PASCAL VOC data set. 
The functions color_map and color_map_viz are taken from 
https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae

The functions color_to_class ans class_to_color use this colormap for the
conversion between a colour on the segmentation mask and a pixelwise
class label for semantic segmentation
"""
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
import cv2

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def color_map_viz():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    nclasses = 21
    row_size = 50
    col_size = 500
    cmap = color_map()
    array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
    array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]

    imshow(array)
    plt.yticks([row_size*i+row_size/2 for i in range(nclasses+1)], labels)
    plt.xticks([])
    plt.show()


def color_to_class(mask_color, num_colors):
    mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
    
    # Define colormap for all classes+background
    cmap = color_map()[:num_colors+1]
    
    # Void colour is set as initial zero value, other colours get index+1
    mask_class = np.zeros((mask_color.shape[0], mask_color.shape[1]))
    for idx, color in enumerate(cmap):
        is_color = cv2.inRange(mask_color, color, color)/255
        mask_class = mask_class + is_color*(idx+1)
    
    return mask_class.reshape((mask_color.shape[0]*mask_color.shape[1],1))

def class_to_color(mask_class, num_colors):    
    # Define colormap and insert void colour at position 0    
    cmap = color_map()[:num_colors+1]
    cmap = np.insert(cmap, 0, [224,224,192], axis=0)
    
    # Convert one hot encoding to pixelwise class label
    mask_class = np.argmax(mask_class, axis=-1)
    
    # Return colour based on pixelwise class label
    mask_color = np.zeros((mask_class.shape[0], 3))
    for idx, color in enumerate(cmap):
        is_class = np.where(mask_class==idx)
        mask_color[is_class] = color

    mask_color = mask_color.reshape((np.int(np.sqrt(mask_class.shape[0])), np.int(np.sqrt(mask_class.shape[0])), 3)).astype(np.uint8)    
    return cv2.cvtColor(mask_color,cv2.COLOR_RGB2BGR)
        
    
    
