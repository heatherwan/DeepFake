#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = "Wanting Lin"

import cv2
import os
import glob

img_dir = 'data_for_extract/input_images'
imagePath = os.path.join(img_dir, '*g')
images = glob.glob(imagePath)
count_img = 0
for img in images:
    image = cv2.imread(img)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        image, scaleFactor=1.3, minNeighbors=3, minSize=(30,30))
    print(f"Found {len(faces)}, Faces!")

    # only face
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        cv2.imwrite(f'data_for_extract/extract/faces/{count_img}-{str(w)}_faces.jpg', roi_color)

    # all frame
    status = cv2.imwrite(f'data_for_extract/extract/all_frame/{count_img}.jpg', image)
    # print(f"Image all_frame/{count_img} written to filesystem: ", status)
    count_img += 1
