#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this is a script to get the fake video result, input the trained model at checkpoint folder and specify the video to generate fake

get the video of (realA) fakeB and (realB) fakeA(frame not ordered)

Following parameters to be determined:
1. --frame: single or double(with original video shows parallel)
2. --audio: yes(keep the origianl audio) or no(create fake video without sound)
Following inputs and models to be determined:
1. --epoch: default as latest
2. -ia: input video name A
3. -ib: input video name B
4. -d: default AtoB, option:BtoA
"""
__author__ = "Wanting_Lin"

import cv2, os, shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--experiment_name', type=str)
parser.add_argument('-epoch', type=str, default='latest')
parser.add_argument('-ia', '--video_namea', type=str)
parser.add_argument('-ib', '--video_nameb', type=str)
parser.add_argument('-d', '--direction', type=str, default='AtoB')

parser.add_argument('--frame',  type=str, default='single')
parser.add_argument('--audio',  type=str, default='no')
args = parser.parse_args()

test_video_pathA = f'datasets/original_video/{args.video_namea}'
test_video_pathB = f'datasets/original_video/{args.video_nameb}'
experiment_name = args.experiment_name
epoch = args.epoch
frame = args.frame
direction = args.direction
audio = args.audio
args.video_namea = f'{args.video_namea.replace(".mp4", "")}_{args.frame}.mp4'
args.video_nameb = f'{args.video_nameb.replace(".mp4", "")}_{args.frame}.mp4'


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    # else:  # delete all in the path
    #     shutil.rmtree(path)
    #     os.mkdir(path)


def main():
    # GET THE DATASET AND CREATE FOLDER
    current_dir = os.getcwd()
    extract_folder = os.path.join(current_dir, f'datasets/{experiment_name}')
    mkdir_if_not_exist(extract_folder)
    extract_folder_testA = os.path.join(extract_folder, 'testA')
    mkdir_if_not_exist(extract_folder_testA)
    extract_folder_testB = os.path.join(extract_folder, 'testB')
    mkdir_if_not_exist(extract_folder_testB)
    fakeA_folder = os.path.join(extract_folder, 'fakeA')
    mkdir_if_not_exist(fakeA_folder)
    fakeB_folder = os.path.join(extract_folder, 'fakeB')
    mkdir_if_not_exist(fakeB_folder)

    # if video not convert to frames
    if len([s for s in os.listdir(extract_folder_testA) if s.endswith('.png')]) == 0:
        extract_video_command = f'ffmpeg -i {test_video_pathA} -vf scale=256:256 {extract_folder_testA}/%05d.png'
        os.system(extract_video_command)
        extract_video_command = f'ffmpeg -i {test_video_pathB} -vf scale=256:256 {extract_folder_testB}/%05d.png'
        os.system(extract_video_command)
    extract_frames_numA = len([s for s in os.listdir(extract_folder_testA) if s.endswith('.png')])
    print('A frame ', extract_frames_numA)
    extract_frames_numB = len([s for s in os.listdir(extract_folder_testB) if s.endswith('.png')])
    print('B frame ', extract_frames_numB)

    # extract audio
    audio_path_A = os.path.join(extract_folder, f"{args.video_namea.replace('mp4', 'mp3')}")
    audio_path_B = os.path.join(extract_folder, f"{args.video_nameb.replace('mp4', 'mp3')}")
    if audio == "yes":
        extract_audio_commandA = f'ffmpeg -i {test_video_pathA} -q:a 0 -map a {audio_path_A}'
        os.system(extract_audio_commandA)
        extract_audio_commandB = f'ffmpeg -i {test_video_pathB} -q:a 0 -map a {audio_path_B}'
        os.system(extract_audio_commandB)

    # RUN TEST.PY TO GET FAKE FRAMES
    input("Press the <ENTER> key to run test.py..")
    if len([s for s in os.listdir(fakeA_folder) if s.endswith('.png')]) == 0:
        run_test_command = f'python test.py --dataroot {extract_folder} --name {experiment_name} --epoch {epoch}\
                                --model cycle_gan --num_test {extract_frames_numA} --direction {direction}'
        os.system(run_test_command)
        copy_result_command = f'cp results/{experiment_name}/test_{epoch}/images/*_fake_A.png {fakeA_folder}'
        print(copy_result_command)
        os.system(copy_result_command)
        copy_result_command = f'cp results/{experiment_name}/test_{epoch}/images/*_fake_B.png {fakeB_folder}'
        print(copy_result_command)
        os.system(copy_result_command)

    # CREATE FAKE VIDEOS
    input("Press the <ENTER> key to run convert video..")
    # realB and fakeA = get fake B based on A
    output_video_wo_audioA = os.path.join(extract_folder, f"{args.video_nameb.replace('.mp4', '_no_soundB.mp4')}")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if frame == 'single':
        out = cv2.VideoWriter(os.path.join(current_dir, output_video_wo_audioA), fourcc, 30.0, (256,256), True)
        # output path, format, frequency, frame size, isColor
        for i in range(extract_frames_numB):
            fake_frame_name = os.path.join(fakeA_folder, ('%05d_fake_A.png' % (i+1)))
            if os.path.exists(fake_frame_name):
                img = cv2.imread(fake_frame_name)
                out.write(img)
                print(f'writing {fake_frame_name}')
            else:
                print(f'file {fake_frame_name} not exist')
        out.release()
    else:
        out = cv2.VideoWriter(os.path.join(current_dir, output_video_wo_audioA), fourcc, 30.0, (512,256), True)
        # output path, format, frequency, frame size, isColor
        for i in range(extract_frames_numB):
            fake_frame_name = os.path.join(fakeA_folder, ('%05d_fake_A.png' % (i+1)))
            real_frame_name = os.path.join(extract_folder_testB, ('%05d.png' % (i+1)))
            if os.path.exists(fake_frame_name) and os.path.exists(real_frame_name):
                fake_img = cv2.imread(fake_frame_name)
                real_img = cv2.imread(real_frame_name)
                img = np.concatenate((real_img, fake_img), axis=1)
                out.write(img)
                print(f'writing {fake_frame_name}')
            else:
                print(f'file {fake_frame_name} not exist')
        out.release()
    print(f'Finished getting fake video B (without sound)')

    # realA and fakeB = get fake A based on B
    output_video_wo_audioB = os.path.join(extract_folder, f"{args.video_namea.replace('.mp4', '_no_soundA.mp4')}")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if frame == 'single':
        out = cv2.VideoWriter(os.path.join(current_dir, output_video_wo_audioB), fourcc, 30.0, (256, 256), True)
        # output path, format, frequency, frame size, isColor
        for i in range(extract_frames_numA):
            fake_frame_name = os.path.join(fakeB_folder, ('%05d_fake_B.png' % (i + 1)))
            if os.path.exists(fake_frame_name):
                img = cv2.imread(fake_frame_name)
                out.write(img)
                print(f'writing {fake_frame_name}')
            else:
                print(f'file {fake_frame_name} not exist')
        out.release()
    else:
        out = cv2.VideoWriter(os.path.join(current_dir, output_video_wo_audioB), fourcc, 30.0, (512, 256), True)
        # output path, format, frequency, frame size, isColor
        for i in range(extract_frames_numA):
            fake_frame_name = os.path.join(fakeB_folder, ('%05d_fake_B.png' % (i + 1)))
            real_frame_name = os.path.join(extract_folder_testA, ('%05d.png' % (i + 1)))
            if os.path.exists(fake_frame_name) and os.path.exists(real_frame_name):
                fake_img = cv2.imread(fake_frame_name)
                real_img = cv2.imread(real_frame_name)
                img = np.concatenate((real_img, fake_img), axis=1)
                out.write(img)
                print(f'writing {fake_frame_name}')
            else:
                print(f'file {fake_frame_name} not exist')
        out.release()
    print(f'Finished getting fake video A (without sound)')

    # add audio
    if audio == "yes":
        output_video_pathA = os.path.join(extract_folder, f"{args.video_namea.replace('.mp4', '_audioB.mp4')}")
        add_audio_comment = f'ffmpeg -i {output_video_wo_audioA} -i {audio_path_A} -map 0 -map 1 -codec copy {output_video_pathA}'
        os.system(add_audio_comment)
        print(f'Finished getting fake video B (with sound)')
        output_video_pathB = os.path.join(extract_folder, f"{args.video_nameb.replace('.mp4', '_audioA.mp4')}")
        add_audio_comment = f'ffmpeg -i {output_video_wo_audioB} -i {audio_path_B} -map 0 -map 1 -codec copy {output_video_pathB}'
        os.system(add_audio_comment)
        print(f'Finished getting fake video A (with sound)')


if __name__ == '__main__':
    main()


