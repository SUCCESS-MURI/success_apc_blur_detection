# author mhatfalv
# splits the CHUK dataset into training (800) and testing (200) images
import argparse
import os
import random
import shutil
import numpy as np
# split up images for creating our dataset

def split_dataset_using_list(args):
    test_list_file = 'dataset/kimetal_test_list.txt'
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
    indexs = np.arange(0,len(images))
    testList = '\t'.join(lines)
    for idx in indexs:
        image = images[idx]
        gt = gtImages[idx]
        if image in testList:
            shutil.move(input_images_file_path + "/" + image, output_testing_images_path + "/" + image)
            shutil.move(input_gt_file_path + "/" + gt, output_testing_gt_path + "/" + gt)
        else:
            shutil.move(input_images_file_path + "/" + image, output_training_images_path + "/" + image)
            shutil.move(input_gt_file_path + "/" + gt, output_training_gt_path + "/" + gt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_input_path', type=str,
                        default='/home/mary/code/local_success_dataset/OrigBlurDataset',
                        help='Path to the CUHK dataset')
    parser.add_argument('--file_output_path', type=str,
                        default='/home/mary/code/local_success_dataset/CUHK_Split_Dataset',
                        help='Output Path to the CUHK datasetfor testing and training')
    args = parser.parse_args()
    split_dataset_using_list(args)