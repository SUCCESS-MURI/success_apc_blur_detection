import argparse
import os
import shutil

import numpy as np
from numpy import random

training_count = 80#36#53
testing_count = 20#9#14

def split_dataset(args):
    # now create folders for correct data vs incorectly labeled data
    path_for_training = os.path.join(args.file_output_path, "Training")
    path_for_testing = os.path.join(args.file_output_path, "Testing")
    os.makedirs(path_for_training)
    os.makedirs(path_for_testing)
    output_testing_images_path = os.path.join(path_for_testing, "images")
    output_training_images_path = os.path.join(path_for_training, "images")
    output_testing_gt_path = os.path.join(path_for_testing, "gt")
    output_training_gt_path = os.path.join(path_for_training, "gt")
    os.makedirs(output_testing_images_path)
    os.makedirs(output_training_images_path)
    os.makedirs(output_training_gt_path)
    os.makedirs(output_testing_gt_path)
    print("Directory for Training data is located in : " + path_for_training)
    print("Directory for Testing data is located in : " + path_for_testing)
    # https://careerkarma.com/blog/python-list-files-in-directory/
    input_images_file_path = os.path.join(args.file_input_path, "image")
    input_gt_file_path = os.path.join(args.file_input_path, "gt")
    images = np.sort(os.listdir(input_images_file_path))
    gtImages = np.sort(os.listdir(input_gt_file_path))
    indexs = np.arange(0,len(images))
    random.shuffle(indexs)
    counttesting = 0
    for idx in indexs:
        image = images[idx]
        gt = gtImages[idx]
        if counttesting < testing_count:
            shutil.move(input_images_file_path + "/"+image, output_testing_images_path + "/"+image)
            shutil.move(input_gt_file_path + "/"+gt, output_testing_gt_path + "/"+gt)
            counttesting += 1
        else:
            shutil.move(input_images_file_path + "/"+image, output_training_images_path + "/"+image)
            shutil.move(input_gt_file_path + "/"+gt, output_training_gt_path + "/"+gt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_input_path', type=str,
                        default='/home/mary/code/local_success_dataset/exposure_dataset_gathering/02_21_2021/brightness',
                        help='Path to the CUHK dataset')
    parser.add_argument('--file_output_path', type=str,
                        default='/home/mary/code/local_success_dataset/02_22_2022_brightness_dataset',
                        help='Output Path to the CUHK datasetfor testing and training')
    args = parser.parse_args()
    split_dataset(args)