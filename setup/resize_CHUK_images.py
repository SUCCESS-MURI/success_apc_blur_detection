# author mhatfalv
# resizes the chuk dataset into 224 by 224
import argparse
import os
import random
import shutil

import cv2
import numpy as np

output_size = (224,224)

def resize_dataset(args):
    # name the input folders
    test_list_file = '../dataset/test_list.txt'
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
    # https://www.pythontutorial.net/python-basics/python-read-text-file/
    with open(test_list_file) as f:
        lines = f.readlines()
    indexs = np.arange(0, len(images))
    testList = '\t'.join(lines)
    for idx in indexs:
        image = images[idx]
        gt = gtImages[idx]
        img = cv2.imread(input_images_file_path + "/" + image)
        # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        resized = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
        img2 = cv2.imread(input_gt_file_path + "/" + gt)
        resized2 = cv2.resize(img2, output_size, interpolation=cv2.INTER_AREA)
        if image in testList:
            cv2.imwrite(output_testing_images_path + "/" + image, resized)
            cv2.imwrite(output_testing_gt_path + "/" + gt, resized2)
        else:
            cv2.imwrite(output_training_images_path + "/" + image, resized)
            cv2.imwrite(output_training_gt_path + "/" + gt, resized2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_input_path', type=str,
                        default='/home/mary/code/local_success_dataset/CUHK_Blur_Detection_Dataset',
                        help='Path to the CUHK dataset size 256 by 256 ')
    parser.add_argument('--file_output_path', type=str,
                        default='/home/mary/code/local_success_dataset/CUHK_Split_Dataset_Resized',
                        help='Output Path to the CUHK dataset for testing and training resized to 224 by 224')
    args = parser.parse_args()
    resize_dataset(args)