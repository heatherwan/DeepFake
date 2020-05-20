#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this is a script to convert video input into images and masks for training the generation model

Following parameters to be determined:
1. Different part of eye can be set from 0 to 255(the higher value means the more training on that part)
2. The ratio of face to other part in image can be set from 1(no difference) to infinite(only learn on face part)
"""
__author__ = 'Wanting Lin'

import os, shutil
import argparse
import CreateMask
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--experiment_name', type=str)
parser.add_argument('-m', '--model_used', type=str)
parser.add_argument('-ia', '--inputA', type=str)
parser.add_argument('-ib', '--inputB', type=str)
parser.add_argument('--continue_train', action='store_true')

args = parser.parse_args()
eye_weight = 255
mouth_weight = 255
nose_weight = 255
face_weight = 255

experiment_name = args.experiment_name
model_name = args.model_used
current_dir = os.getcwd()
print(current_dir)
main_experiment_path = os.path.join(current_dir, f'datasets/{experiment_name}')
trainA_video_path = os.path.join(f'datasets/original_video', f'{args.inputA}')
trainB_video_path = os.path.join(f'datasets/original_video', f'{args.inputB}')


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:  # delete all in the path
        shutil.rmtree(path)
        os.mkdir(path)


def main():
    # set output frame folder
    mkdir_if_not_exist(main_experiment_path)
    trainA_folder = os.path.join(main_experiment_path, f'trainA')
    print(trainA_folder)
    mkdir_if_not_exist(trainA_folder)
    trainB_folder = os.path.join(main_experiment_path, f'trainB')
    mkdir_if_not_exist(trainB_folder)
    maskA_folder = os.path.join(main_experiment_path, f'maskA')
    mkdir_if_not_exist(maskA_folder)
    maskB_folder = os.path.join(main_experiment_path, f'maskB')
    mkdir_if_not_exist(maskB_folder)

    # extract frames from video
    extract_A_command = f'ffmpeg -i {trainA_video_path} -vf scale=256:256 {trainA_folder}/%05d.png'
    extract_B_command = f'ffmpeg -i {trainB_video_path} -vf scale=256:256 {trainB_folder}/%05d.png'
    os.system(extract_A_command)
    os.system(extract_B_command)
    extract_frames_num_A = len([s for s in os.listdir(trainA_folder) if s.endswith('.png')])
    extract_frames_num_B = len([s for s in os.listdir(trainB_folder) if s.endswith('.png')])

    # get image mask
    mask_creater = CreateMask.CreateMask(face_weight, mouth_weight, nose_weight, eye_weight)
    for i in range(extract_frames_num_A):
        print(f' {i+1} / {extract_frames_num_A} frame A mask processed')
        imageA = os.path.join(trainA_folder, ('%05d.png' % (i + 1)))
        saveA = os.path.join(maskA_folder, ('%05d.png' % (i + 1)))
        maskA = mask_creater.create_mask(imageA)
        cv2.imwrite(saveA, maskA)

    for i in range(extract_frames_num_B):
        print(f' {i + 1} / {extract_frames_num_B} frame B mask processed')
        imageB = os.path.join(trainB_folder, ('%05d.png' % (i + 1)))
        saveB = os.path.join(maskB_folder, ('%05d.png' % (i + 1)))
        maskB = mask_creater.create_mask(imageB)
        cv2.imwrite(saveB, maskB)

    # training
    training_command = f'python train.py --dataroot {main_experiment_path} --name {experiment_name} \
                    --model {model_name} --pool_size 50 --no_dropout --face_mask --face_weight 10.0'
    if args.continue_train:
        training_command += ' --continue_train'
    print(training_command)
    os.system(training_command)


if __name__ == '__main__':
    main()

