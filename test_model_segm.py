# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:33:15 2020

@author: astri
"""

#%% Import modules
import os
import random
import numpy as np

from ReadData import get_JPEGimages, get_PNGsegments
from VOClabelcolormap import color_to_class, class_to_color
from keras.utils import to_categorical
from  tensorflow.keras.preprocessing.image import apply_affine_transform as transform

from keras import backend as K
from keras_segmentation.models.model_utils import get_segmentation_model

from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras import optimizers

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

# #%% Define loss function based on dice coefficient
# def dice_coef(y_true, y_pred, smooth=1):
    
#     y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=22)[...,1:])
#     y_pred_f = K.flatten(y_pred[...,1:])
#     intersect = K.sum(y_true_f * y_pred_f, axis=-1)
#     denom = K.sum(y_true_f + y_pred_f, axis=-1)
#     return K.mean((2. * intersect / (denom + smooth)))

# def dice_coef_loss(y_true, y_pred):
#     '''
#     Dice loss to minimize. Pass to model as loss during compile statement
#     '''
#     return 1 - dice_coef(y_true, y_pred)

#%% Define generators for data augmentation and batch creation
def train_data_gen(filelist, batch_size, num_classes):
    # Read the files from the list in the folder
    current_path = os.getcwd()
    folder = os.path.join(current_path,
                                  'VOCtrainval_11-May-2009',
                                  'VOCdevkit',
                                  'VOC2009')

    # Read the data and write it to an array
    segm_resize = True      # resize images to 256x256
    segm_img = get_JPEGimages(filelist, folder, resize=segm_resize)
    segm_mask = get_PNGsegments(filelist, folder, resize=segm_resize).astype(np.uint8)
    
    # Create batches
    c = 0  
    while(True):        
        images = np.zeros((batch_size, 256, 256, 3)).astype('float')
        masks = np.zeros((batch_size, 256*256, num_classes+2)).astype('float')
    
        for i in range(c, min(c+batch_size,len(segm_img))):
            # Each original image is added to the training data
            images[i-c] = segm_img[i]        
            img_mask = color_to_class(segm_mask[i], num_classes)
            masks[i-c] = to_categorical(img_mask, num_classes=num_classes+2)
        
            # # Next to this each image is either rotated or flipped and added to the training data
            # if random.random() > 0.5:
            #     img = transform(segm_img[i], theta=90)
            #     img_mask = transform(segm_mask[i], theta=90, order=1)
            # else:
            #     img = np.flip(segm_img[i], axis=1)
            #     img_mask = np.flip(segm_mask[i], axis=1)
              
            # images[i-c+np.int(batch_size/2)] = img         
            # img_mask = color_to_class(img_mask, num_classes)
            # masks[i-c+np.int(batch_size/2)] = to_categorical(img_mask, num_classes=num_classes+2)
    
        c+=batch_size
    
        yield images, masks
        
def val_data_gen(filelist, batch_size, num_classes):
    # Read the files from the list in the folder
    current_path = os.getcwd()
    folder = os.path.join(current_path,
                                  'VOCtrainval_11-May-2009',
                                  'VOCdevkit',
                                  'VOC2009')
    
    # Read the data and write it to an array
    segm_resize = True
    segm_img = get_JPEGimages(filelist, folder, resize=segm_resize)
    segm_mask = get_PNGsegments(filelist, folder, resize=segm_resize).astype(np.uint8)
    
    # Create batches
    c = 0  
    while(True):        
        images = np.zeros((batch_size, 256, 256, 3)).astype('float')
        masks = np.zeros((batch_size, 256*256, num_classes+2)).astype('float')
    
        for i in range(c, min(c+batch_size,len(segm_img))):
            # Each original image is added to the validation or test data
            images[i-c] = segm_img[i]        
            img_mask = color_to_class(segm_mask[i], num_classes)
            masks[i-c] = to_categorical(img_mask, num_classes=num_classes+2)

        c+=batch_size
        
        yield images, masks
        

#%% Load data
batch_size = 32
num_classes = 20    # this is the actual number of classes, void and backgournd are added later

train_gen = train_data_gen('train_segm.txt', batch_size, num_classes)
val_gen = val_data_gen('val_segm.txt', batch_size, num_classes)
#test_gen = val_data_gen('segm_test.txt', batch_size, num_classes)

print('LOADING DATA FINISHED')

#%% Define neural network
img_input = Input(shape=(256,256,3))
### AANPASSEN kernel_initializer = 'he_normal'
# Block 1
conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(img_input)
conv1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(conv1)
drop1 = Dropout(0.5)(pool1)
# Block 2
conv2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(conv2)
drop2 = Dropout(0.5)(pool2)
# Block 3
conv3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(drop2)
conv3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
conv3 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(conv3)
drop3 = Dropout(0.5)(pool3)
# Block 4
conv4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(drop3)
conv4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
conv4 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
pool4 = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(conv4)
drop4 = Dropout(0.5)(pool4)
# Block 5
conv5 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(drop4)
conv5 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)
conv5 = Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)
pool5 = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(conv5)
drop5 = Dropout(0.5)(pool5)
# Block 6
up6 = Conv2DTranspose(512, (3,3), strides=(2,2), activation='relu', padding='same',  kernel_initializer = 'he_normal')(conv5)
up6 = concatenate([up6, conv4], axis=-1)
drop6 = Dropout(0.2)(up6)
conv6 = Conv2D(512, (3,3), activation='relu', padding='same',  kernel_initializer = 'he_normal')(drop6)
conv6 = Conv2D(512, (3,3), activation='relu', padding='same',  kernel_initializer = 'he_normal')(conv6)
# Block 7
up7 = Conv2DTranspose(256, (3,3), strides=(2,2), activation='relu', padding='same',  kernel_initializer = 'he_normal')(conv6)
up7 = concatenate([up7, conv3], axis=-1)
drop7 = Dropout(0.2)(up7)
conv7 = Conv2D(256, (3,3), activation='relu', padding='same',  kernel_initializer = 'he_normal')(drop7)
conv7 = Conv2D(256, (3,3), activation='relu', padding='same',  kernel_initializer = 'he_normal')(conv7)
# Block 8
up8 = Conv2DTranspose(128, (3,3), strides=(2,2), activation='relu', padding='same',  kernel_initializer = 'he_normal')(conv7)
up8 = concatenate([up8, conv2], axis=-1)
drop8 = Dropout(0.2)(up8)
conv8 = Conv2D(128, (3,3), activation='relu', padding='same',  kernel_initializer = 'he_normal')(drop8)
conv8 = Conv2D(128, (3,3), activation='relu', padding='same',  kernel_initializer = 'he_normal')(conv8)
# Block 9
up9 = Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same')(conv8)
up9 = concatenate([up9, conv1], axis=-1)
drop9 = Dropout(0.2)(up9)
conv9 = Conv2D(64, (3,3), activation='relu', padding='same',  kernel_initializer = 'he_normal')(drop9)
conv9 = Conv2D(64, (3,3), activation='relu', padding='same',  kernel_initializer = 'he_normal')(conv9)

out = Conv2D(num_classes+2, (1, 1) , padding='same',  kernel_initializer = 'he_normal')(conv9)

#%% Compile neural network
model = get_segmentation_model(img_input, out) # this would build the segmentation model

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

print('COMPILING MODEL FINISHED')
#%% Train neural network
current_path = os.getcwd()
weights_path  = os.path.join(r'C:\Users\astri\Documenten\__5e jaar\Vision\Project', 'segmentation')

# Create of list of callbacks for intermediate results
model_names = weights_path + '\weights.{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(model_names, monitor='val_acc', 
                              verbose=1, save_best_only=True, mode='max')

#os.makedirs(weights_path+'\log.out', mode=0o777, exist_ok=True)
csv_logger = CSVLogger(weights_path+'\log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor='val_acc', verbose=1,
                              min_delta=0.01, patience=10, mode='max')
callbacks_list = [checkpoint, csv_logger, earlystopping]

# Train model using training and validation generators
results = model.fit_generator(train_gen,
                          epochs=10, 
                          steps_per_epoch=24,
                          validation_data=val_gen, 
                          validation_steps=24, 
                          callbacks=callbacks_list)
model.save('Model.h5')

print('TRAINING MODEL FINISHED')

#%% Predictions for test data
acc = model.predict_generator(val_gen, steps=10)
pred_val = model.predict_generator(val_gen, steps=10)

pred_val_masks = []
for mask in enumerate(pred_val):
    mask_color = class_to_color(mask, num_classes)
    pred_val_masks.append(mask_color)


