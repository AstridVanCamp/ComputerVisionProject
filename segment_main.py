# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:39:40 2020

@author: astri
"""
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from keras import layers
from keras import models
from keras import optimizers
from keras_segmentation.models.unet import vgg_unet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from ReadData import get_JPEGimages, get_PNGsegments
from VOClabelcolormap import one_hot_encode

from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=21)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)

## Load data
# current path is fetched
current_path = os.getcwd()
# folder is fetched where images are located dynamically
folder = os.path.join(current_path,
                             'VOCtrainval_11-May-2009',
                             'VOCdevkit',
                             'VOC2009')
segm_resize = True
segm_train_img = get_JPEGimages('segm_train.txt', folder, resize=segm_resize)
segm_val_img = get_JPEGimages('segm_val.txt', folder, resize=segm_resize)
segm_test_img = get_JPEGimages('segm_test.txt', folder, resize=segm_resize)

segm_train_class = get_PNGsegments('segm_train.txt', folder, resize=segm_resize)
segm_val_class = get_PNGsegments('segm_val.txt', folder, resize=segm_resize)
segm_test_class = get_PNGsegments('segm_test.txt', folder, resize=segm_resize)

train_masks = one_hot_encode(segm_train_class, 20)
val_masks = one_hot_encode(segm_val_class, 20)
test_masks = one_hot_encode(segm_test_class, 20)

## Create batches (and data augmentation)
batch_size = 32

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_img_gen = train_datagen.flow(segm_train_img, batch_size=batch_size)
train_class_gen = train_datagen.flow(segm_train_class, batch_size=batch_size)
train_generator = zip(train_img_gen, train_class_gen)

val_datagen = ImageDataGenerator(rescale=1./255)
val_img_gen = val_datagen.flow(segm_val_img, batch_size=batch_size)
val_class_gen =val_datagen.flow(segm_val_class, batch_size=batch_size)
val_generator = zip(val_img_gen, val_class_gen)

NO_OF_TRAINING_IMAGES = len(segm_train_img)
NO_OF_VAL_IMAGES = len(segm_val_img)

del segm_train_img, segm_train_class, segm_val_img, segm_val_class

# ## Define neural network
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(375,500,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

model = vgg_unet(n_classes=20, input_height=375, input_width=500)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# NO_OF_EPOCHS = 10
# BATCH_SIZE = batch_size

# weights_path  = os.path.join(current_path, 'segmentation', 'weights_scratch')
# checkpoint = ModelCheckpoint(weights_path, monitor='acc', 
#                              verbose=1, save_best_only=True, mode='max')
# csv_logger = CSVLogger('./log.out', append=True, separator=';')
# earlystopping = EarlyStopping(monitor='acc', verbose=1,
#                               min_delta=0.01, patience=3, mode='max')
# callbacks_list = [checkpoint, csv_logger, earlystopping]

# results = model.fit_generator(train_generator,
#                           epochs=NO_OF_EPOCHS, 
#                           steps_per_epoch=(NO_OF_TRAINING_IMAGES//BATCH_SIZE),
#                           validation_data=val_generator, 
#                           validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
#                           callbacks=callbacks_list)
# model.save('Model.h5')

