# author mhatfalv
# resizes the new dataset into 256 by 256
import argparse
import os
import cv2
import numpy as np

output_size = (256,256,3)

def resize_dataset(args):
    # name the input folders
    # now create folders for correct data vs incorrectly labeled data
    #path_for_training = os.path.join(args.file_output_path, "Training")
    os.makedirs(args.file_output_path)
    output_training_images_path = os.path.join(args.file_output_path, "image")
    output_training_gt_path = os.path.join(args.file_output_path, "gt")
    os.makedirs(output_training_images_path)
    os.makedirs(output_training_gt_path)
    print("Directory for data is located in : " + args.file_output_path)
    # https://careerkarma.com/blog/python-list-files-in-directory/
    input_images_file_path = os.path.join(args.file_input_path, "images")
    input_gt_file_path = os.path.join(args.file_input_path, "gt")
    images = np.sort(os.listdir(input_images_file_path))
    gtImages = np.sort(os.listdir(input_gt_file_path))
    indexs = np.arange(0, len(images))
    for idx in indexs:
        image = images[idx]
        gt = gtImages[idx]
        img = cv2.imread(input_images_file_path + "/" + image)
        # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        resized = np.zeros(output_size)
        resized[16:16+224,16:16+224,:] = img
        img2 = cv2.imread(input_gt_file_path + "/" + gt)
        resized2 = np.zeros(output_size)
        resized2[16:16+224,16:16+224,:] = img2
        cv2.imwrite(output_training_images_path + "/" + image, resized)
        cv2.imwrite(output_training_gt_path + "/" + gt, resized2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_input_path', type=str,
                        default='/home/mary/code/local_success_dataset/ssc_dataset_training_2',
                        help='Path to the dataset size 224 by 224 ')
    parser.add_argument('--file_output_path', type=str,
                        default='/home/mary/code/local_success_dataset/BlurDetectionDataset/Training_Resized_2',
                        help='Output Path to the dataset for training resized to 256 by 256')
    args = parser.parse_args()
    resize_dataset(args)