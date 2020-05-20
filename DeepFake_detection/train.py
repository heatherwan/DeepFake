#!/usr/bin/env python
#  -*- coding:utf-8 -*-
__author__ = "Wanting_Lin"

from classifiers import *
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
experiment_name = 'trainv4onv3'
IMGWIDTH = 256
chose_pretrained_model = 'weights/trainv3onv2.h5'

train_steps = len([s for s in os.listdir('datasets/train_images/fake') if s.endswith('.jpg')])
valid_steps = len([s for s in os.listdir('datasets/valid_images/fake') if s.endswith('.jpg')])
print(train_steps)
print(valid_steps)
# load training data
dataGenerator = ImageDataGenerator(rescale=1./255)
train_generator = dataGenerator.flow_from_directory(
    'datasets/train_images',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary',
    subset='training',
    seed=42)
print(f'class indices: {train_generator.class_indices}')  # df:0 real:1
test_generator = dataGenerator.flow_from_directory(
    'datasets/valid_images',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary',
    subset='training',
    seed=42)

# load model
classifier = Meso4(experiment_name)
print('finish load model')
# load pre-trained model
classifier.load(chose_pretrained_model)
print('finish load model')

result = classifier.fit(train_generator, test_generator, train_steps, valid_steps)
