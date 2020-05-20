#!/usr/bin/env python
#  -*- coding:utf-8 -*-
__author__ = "Wanting_Lin"

from classifiers import *
# from pipeline import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
experiment_name = 'trainv3onv2'
# 1 - Load the model and its pretrained weights
classifier = Meso4(experiment_name)
classifier.load(f'weights/{experiment_name}.h5')

# 2 - Image generator
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'datasets/test_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training',
        shuffle=False,
        seed=42)

# 3 - Prediction of frames
# overall prediction result
result = classifier.get_accuracy(generator)
print(f'Test Loss: {result[0]}; Test Accuracy: {result[1]}')

# Prediction result for each frame
actual = generator.classes
pred = classifier.predict(generator)
name = generator.filenames
incorrect_classification = open(f'output/incorrect_test_images_{experiment_name}', 'w')
incorrect_classification.write(f'prediction of ... is incorrect :\n name\treal\tpred\n')

for x, y, z in zip(actual, pred, name):
    if abs(x-y) > 0.5:
        # print(f'prediction of {z} is not correct')
        incorrect_classification.write(f'{z}\t{x}\t{y}\n')

incorrect_classification.close()

print(confusion_matrix(actual, pred>0.5))

#
# # 4 - Prediction for a video dataset
# classifier.load('weights/Meso4_F2F')
# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])