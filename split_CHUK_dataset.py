# author mhatfalv
# splits the CHUK dataset into training (800) and testing (200) images
import argparse
import os
import random
import shutil

import numpy as np
num_of_motion = 60
num_of_defocus = 140

def split_dataset(args):
    # now create folders for correct data vs incorectly labeled data
    path_for_training = os.path.join(args.file_output_path, "Training")
    path_for_testing = os.path.join(args.file_output_path, "Testing")
    os.makedirs(path_for_training)
    os.makedirs(path_for_testing)
    output_testing_images_path = os.path.join(path_for_testing, "image")
    output_training_images_path = os.path.join(path_for_training, "image")
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
    countMotion = 0
    countFocus = 0
    for idx in indexs:
        image = images[idx]
        gt = gtImages[idx]
        if "motion" in image and countMotion < num_of_motion:
            shutil.move(input_images_file_path + "/"+image, output_testing_images_path + "/"+image)
            shutil.move(input_gt_file_path + "/"+gt, output_testing_gt_path + "/"+gt)
            countMotion += 1
        elif "focus" in image and countFocus < num_of_defocus:
            shutil.move(input_images_file_path + "/"+image, output_testing_images_path +"/"+image)
            shutil.move(input_gt_file_path + "/"+gt, output_testing_gt_path + "/"+gt)
            countFocus += 1
        else:
            shutil.move(input_images_file_path + "/"+image, output_training_images_path + "/"+image)
            shutil.move(input_gt_file_path + "/"+gt, output_training_gt_path + "/"+gt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_input_path', type=str,
                        default='/home/mary/code/local_success_dataset/CUHK_Blur_Detection_Dataset',
                        help='Path to the CUHK dataset')
    parser.add_argument('--file_output_path', type=str,
                        default='/home/mary/code/local_success_dataset/CUHK_Split_Dataset',
                        help='Output Path to the CUHK datasetfor testing and training')
    args = parser.parse_args()
    split_dataset(args)