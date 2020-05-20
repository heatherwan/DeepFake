#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this is a script to get mask object from each image

"""
__author__ = 'Wanting Lin'

import face_alignment
import cv2
from collections import OrderedDict
import numpy as np


class CreateMask(object):
    def __init__(self, face_weight, mouth_weight, nose_weight, eye_weight):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.facial_landmark_idx = OrderedDict([
            ("face", (0, 68, face_weight)),
            ("mouth", (48, 68, mouth_weight)),
            ("nose", (27, 35, nose_weight)),
            ("right_eye", (36, 42, eye_weight)),
            ("left_eye", (42, 48, eye_weight))
        ])

    def create_mask(self, image_path):
        # read image and convert
        image = cv2.imread(image_path)
        # image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        # get facial landmark
        preds = self.fa.get_landmarks(image)
        if preds is not None:
            pred = preds[0]
            mask = np.zeros_like(image)  # create same size black binary mask
            # draw part mask with weight
            for name, (start, end, weight) in self.facial_landmark_idx.items():
                # Draw face shape mask
                # print(f'draw {name}')
                pnts = [(pred[i, 0], pred[i, 1]) for i in range(start, end)]
                hull = cv2.convexHull(np.array(pnts)).astype(np.int32)
                mask = cv2.drawContours(mask, [hull], -1, (weight, weight, weight), -1)
            mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)  # increase line width
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
        else:
            print(f'no face found in {image_path}, return black mask')
            mask = np.zeros_like(image)
        return mask
