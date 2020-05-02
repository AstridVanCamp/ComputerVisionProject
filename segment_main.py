# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:39:40 2020

@author: astri
"""
#%%
import os
import random
import numpy as np

from ReadData import get_JPEGimages, get_PNGsegments
from VOClabelcolormap import unique_class
from keras.utils import to_categorical
from  tensorflow.keras.preprocessing.image import apply_affine_transform as transform

from keras import backend as K
from keras_segmentation.models.model_utils import get_segmentation_model

from keras import layers
from keras import optimizers

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

#%%
def dice_coef(y_true, y_pred, smooth=1):
    
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=22)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)

#%%
def train_data_gen(filelist, batch_size, num_classes):
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
    while(True):        
        images = np.zeros((batch_size, 256, 256, 3)).astype('float')
        masks = np.zeros((batch_size, 256*256, num_classes+2)).astype('float')
    
        for i in range(c, min(c+np.int(batch_size/2),len(segm_img))): #initially from 0 to 16, c = 0.  
            # Each original image is added to the training data
            images[i-c] = segm_img[i]        
            img_mask = unique_class(segm_mask[i], num_classes)
            masks[i-c] = to_categorical(img_mask, num_classes=num_classes+2)
        
            # Next to this each image is either rotated or flipped and added to the training data
            if random.random() > 0.5:
                img = transform(segm_img[i], theta=90)
                img_mask = transform(segm_mask[i], theta=90)
            else:
                img = np.flip(segm_img[i], axis=1)
                img_mask = np.flip(segm_mask[i], axis=1)
              
            images[i-c+np.int(batch_size/2)] = img         
            img_mask = unique_class(img_mask, num_classes)
            masks[i-c+np.int(batch_size/2)] = to_categorical(img_mask, num_classes=num_classes+2)
    
        c+=np.int(batch_size/2)
    
        yield images, masks
        
def val_data_gen(filelist, batch_size, num_classes):
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
    while(True):        
        images = np.zeros((batch_size, 256, 256, 3)).astype('float')
        masks = np.zeros((batch_size, 256*256, num_classes+2)).astype('float')
    
        for i in range(c, min(c+batch_size,len(segm_img))): #initially from 0 to 16, c = 0.  
            # Each original image is added to the training data
            images[i-c] = segm_img[i]        
            img_mask = unique_class(segm_mask[i], num_classes)
            masks[i-c] = to_categorical(img_mask, num_classes=num_classes+2)

        c+=batch_size
    
        yield images, masks
        

#%% Load data
batch_size = 16
num_classes = 20

train_gen = train_data_gen('segm_train.txt', batch_size, num_classes)
val_gen = val_data_gen('segm_val.txt', batch_size, num_classes)
test_gen = val_data_gen('segm_test.txt', batch_size, num_classes)

#%% Define neural network
img_input = layers.Input(shape=(256,256,3))

conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(img_input)
conv1 = layers.Dropout(0.2)(conv1)
conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
pool1 = layers.MaxPooling2D((2,2))(conv1)

conv2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
conv2 = layers.Dropout(0.2)(conv2)
conv2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
pool2 = layers.MaxPooling2D((2,2))(conv2)

conv3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
conv3 = layers.Dropout(0.2)(conv3)
conv3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv3)

up1 = layers.concatenate([layers.UpSampling2D((2, 2))(conv3), conv2], axis=-1)
conv4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(up1)
conv4 = layers.Dropout(0.2)(conv4)
conv4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv4)

up2 = layers.concatenate([layers.UpSampling2D((2, 2))(conv2), conv1], axis=-1)
conv5 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(up2)
conv5 = layers.Dropout(0.2)(conv5)
conv5 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(conv5)

out = layers.Conv2D(num_classes+2, (1, 1) , padding='same')(conv5)

#%%
model = get_segmentation_model(img_input, out) # this would build the segmentation model

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
#%%
current_path = os.getcwd()
weights_path  = os.path.join(current_path, 'segmentation', 'weights_scratch')

model_names = weights_path + '\weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(model_names, monitor='acc', 
                              verbose=1, save_best_only=True, mode='max')

#os.makedirs(weights_path+'\log.out', mode=0o777, exist_ok=True)
csv_logger = CSVLogger(weights_path+'\log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor='acc', verbose=1,
                              min_delta=0.01, patience=3, mode='max')
callbacks_list = [checkpoint, csv_logger, earlystopping]

results = model.fit_generator(train_gen,
                          epochs=10, 
                          steps_per_epoch=3,
                          validation_data=val_gen, 
                          validation_steps=2, 
                          callbacks=callbacks_list)
model.save('Model.h5')

