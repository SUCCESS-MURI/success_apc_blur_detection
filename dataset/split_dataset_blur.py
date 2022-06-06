import argparse
import os
import shutil

import numpy as np
from numpy import random

training_count = 80
testing_count = 10
validation_count = 10

def split_dataset(args):
    # now create folders for correct data vs incorectly labeled data
    path_for_training = os.path.join(args.file_output_path, "Training")
    path_for_testing = os.path.join(args.file_output_path, "Testing")
    path_for_validation = os.path.join(args.file_output_path, "Validation")
    os.makedirs(path_for_training)
    os.makedirs(path_for_testing)
    output_testing_normal_path = os.path.join(path_for_testing, "normal")
    output_training_normal_path = os.path.join(path_for_training, "normal")
    output_validation_normal_path = os.path.join(path_for_validation, "normal")
    output_testing_focus_path = os.path.join(path_for_testing, "focus")
    output_training_focus_path = os.path.join(path_for_training, "focus")
    output_validation_focus_path = os.path.join(path_for_validation, "focus")
    output_testing_motion_path = os.path.join(path_for_testing, "motion")
    output_training_motion_path = os.path.join(path_for_training, "motion")
    output_validation_motion_path = os.path.join(path_for_validation, "motion")
    output_testing_overexposure_path = os.path.join(path_for_testing, "overexposure")
    output_training_overexposure_path = os.path.join(path_for_training, "overexposure")
    output_validation_overexposure_path = os.path.join(path_for_validation, "overexposure")
    output_testing_underexposure_path = os.path.join(path_for_testing, "underexposure")
    output_training_underexposure_path = os.path.join(path_for_training, "underexposure")
    output_validation_underexposure_path = os.path.join(path_for_validation, "underexposure")
    os.makedirs(output_training_normal_path)
    os.makedirs(output_testing_normal_path)
    os.makedirs(output_validation_normal_path)
    os.makedirs(output_testing_focus_path)
    os.makedirs(output_training_focus_path)
    os.makedirs(output_validation_focus_path)
    os.makedirs(output_training_motion_path)
    os.makedirs(output_testing_motion_path)
    os.makedirs(output_validation_motion_path)
    os.makedirs(output_testing_overexposure_path)
    os.makedirs(output_training_overexposure_path)
    os.makedirs(output_validation_overexposure_path)
    os.makedirs(output_testing_underexposure_path)
    os.makedirs(output_training_underexposure_path)
    os.makedirs(output_validation_underexposure_path)
    print("Directory for Training data is located in : " + path_for_training)
    print("Directory for Testing data is located in : " + path_for_testing)
    # https://careerkarma.com/blog/python-list-files-in-directory/
    images = sorted(os.listdir(args.file_input_path),key=lambda x: int(x.split('_')[0]))
    indexs = np.arange(0,100)
    random.shuffle(indexs)
    counttesting = 0
    countvalidation = 0
    for idx in indexs:
        imageset = images[idx*5:idx*5+5]
        imageset = sorted(imageset,key=lambda x: x.split('_')[2])
        if counttesting < testing_count:
            shutil.move(args.file_input_path + "/"+imageset[2], output_testing_normal_path + "/"+imageset[2])
            shutil.move(args.file_input_path + "/"+imageset[1], output_testing_motion_path + "/"+imageset[1])
            shutil.move(args.file_input_path + "/" + imageset[0], output_testing_focus_path + "/" + imageset[0])
            shutil.move(args.file_input_path + "/" + imageset[3], output_testing_overexposure_path + "/" + imageset[3])
            shutil.move(args.file_input_path + "/" + imageset[4], output_testing_underexposure_path + "/" + imageset[4])
            counttesting += 1
        elif countvalidation < validation_count:
            shutil.move(args.file_input_path + "/" + imageset[2], output_validation_normal_path + "/" + imageset[2])
            shutil.move(args.file_input_path + "/" + imageset[1], output_validation_motion_path + "/" + imageset[1])
            shutil.move(args.file_input_path + "/" + imageset[0], output_validation_focus_path + "/" + imageset[0])
            shutil.move(args.file_input_path + "/" + imageset[3], output_validation_overexposure_path + "/" +
                        imageset[3])
            shutil.move(args.file_input_path + "/" + imageset[4], output_validation_underexposure_path + "/" +
                        imageset[4])
            countvalidation += 1
        else:
            shutil.move(args.file_input_path + "/" + imageset[2], output_training_normal_path + "/" + imageset[2])
            shutil.move(args.file_input_path + "/" + imageset[1], output_training_motion_path + "/" + imageset[1])
            shutil.move(args.file_input_path + "/" + imageset[0], output_training_focus_path + "/" + imageset[0])
            shutil.move(args.file_input_path + "/" + imageset[3], output_training_overexposure_path + "/" + imageset[3])
            shutil.move(args.file_input_path + "/" + imageset[4], output_training_underexposure_path + "/" +
                        imageset[4])

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