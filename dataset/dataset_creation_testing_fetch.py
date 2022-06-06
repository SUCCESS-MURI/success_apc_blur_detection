# Author mhatfalv
# Create the blur dataset
import argparse
import copy
import glob
import random

import cv2
import imageio
import numpy as np
import skimage.color
from skimage.color import rgb2gray
from skimage.morphology import disk
from scipy.signal import convolve2d

from PIL import Image as im, ImageEnhance
import tensorlayer as tl
import matplotlib.pyplot as plt

# create motion blur
# https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
from utils import read_all_imgs

# create testing dataset for chuk with brightness and darkness images
def create_dataset_for_testing(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_normal_list = sorted(tl.files.load_file_list(path=args.data_dir +'/normal', regx='/*.(png|PNG)',
                                                        printable=False))
    images_focus_list = sorted(tl.files.load_file_list(path=args.data_dir +'/focus', regx='/*.(png|PNG)',
                                                       printable=False))
    images_motion_list = sorted(tl.files.load_file_list(path=args.data_dir +'/motion', regx='/*.(png|PNG)',
                                                        printable=False))
    images_overexposure_list = sorted(tl.files.load_file_list(path=args.data_dir +'/overexposure', regx='/*.(png|PNG)',
                                                              printable=False))
    images_underexposure_list = sorted(tl.files.load_file_list(path=args.data_dir +'/underexposure',
                                                               regx='/*.(png|PNG)', printable=False))
    images_sailency_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency', regx='/*.(png|PNG)',
                                                        printable=False))
    images_sailency_exposed_list = sorted(tl.files.load_file_list(path=args.salinecy_exposed_data_dir,
                                                                  regx='/*.(png|PNG)',printable=False))
    # these images have out of focus blur already we need to create motion blur, and the over and under exposure images
    imagesNOrigonal = read_all_imgs(images_normal_list, path=args.data_dir +'/normal/', n_threads=100, mode='RGB')
    imagesFOrigonal = read_all_imgs(images_focus_list, path=args.data_dir +'/focus/', n_threads=100, mode='RGB')
    imagesMOrigonal = read_all_imgs(images_motion_list, path=args.data_dir + '/motion/', n_threads=100, mode='RGB')
    imagesOOrigonal = read_all_imgs(images_overexposure_list, path=args.data_dir + '/overexposure/', n_threads=100,
                                    mode='RGB')
    imagesUOrigonal = read_all_imgs(images_underexposure_list, path=args.data_dir + '/underexposure/',
                                    n_threads=100, mode='RGB')
    imagesSaliency = read_all_imgs(images_sailency_list, path=args.data_dir + '/saliency/',
                                    n_threads=100, mode='GRAY')
    imagesESaliency = read_all_imgs(images_sailency_exposed_list, path=args.salinecy_exposed_data_dir + '/', n_threads=100,
                                    mode='GRAY')

    # save normal images
    # 1st need to get the saliency image
    # 2nd need to make focus pixels background
    # 3rd need to save normal image and gt
    for i in range(len(imagesNOrigonal)):
        image = imagesNOrigonal[i]
        saveName = args.output_data_dir + "/images/"+ str(i) + "_" + args.data_extension
        imageio.imsave(saveName, image)
        # focus blur default for 0 pixels
        saliencyBMask = imagesSaliency[i]
        (T, saliency) = cv2.threshold(saliencyBMask, .001, 1, cv2.THRESH_BINARY)
        saliency[saliency == 0] = 128  # focus
        saliency[saliency == 1] = 0 # normal
        saveName = args.output_data_dir + "/gt/"+str(i) + "_" + args.data_extension
        cv2.imwrite(saveName, saliency)

    # save motion images
    # 1st need to get the motion image
    # 2nd need to save motion image and gt
    j = len(imagesNOrigonal)
    for i in range(len(imagesMOrigonal)):
        image = imagesMOrigonal[i]
        saveName = args.output_data_dir + "/images/"+ str(j) + "_" + args.data_extension
        imageio.imsave(saveName, image)
        # motion blur
        saliencyBMask = np.ones(final_shape)*64
        saveName = args.output_data_dir + "/gt/"+str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliencyBMask)
        j += 1

    # save focus images
    # 1st need to get the focus image
    # 2nd need to save focus image and gt
    for i in range(len(imagesFOrigonal)):
        image = imagesFOrigonal[i]
        saveName = args.output_data_dir + "/images/" + str(j) + "_" + args.data_extension
        imageio.imsave(saveName, image)
        # motion blur
        saliencyBMask = np.ones(final_shape) * 128
        saveName = args.output_data_dir + "/gt/" + str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliencyBMask)
        j += 1

    # save overexposed images
    # 1st need to get the s![](../../local_success_dataset/fetch_images/output_blur_data_06_20_2022/Blur_Detection_Input/Testing/gt/36_.png)aliency image
    # 2nd need to make focus pixels background
    # 3rd need to save overexposed image and gt
    for i in range(len(imagesOOrigonal)):
        image = imagesOOrigonal[i]
        saveName = args.output_data_dir + "/images/"+ str(j) + "_" + args.data_extension
        imageio.imsave(saveName, image)
        # focus blur default for 0 pixels
        saliencyBMask = imagesESaliency[i]
        (T, saliency) = cv2.threshold(saliencyBMask, .1, 1, cv2.THRESH_BINARY)
        imagegray = skimage.color.rgb2gray(image)*255
        saliency[saliency == 0] = 128  # focus
        saliency[saliency == 1] = 255  # overexposure
        saliency[imagegray < 45] = 192
        saliency[imagegray > 225] = 255
        saveName = args.output_data_dir + "/gt/"+str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliency)
        j += 1

    # save underexposed images
    # 1st need to get the saliency image
    # 2nd need to make focus pixels background
    # 3rd need to save underexposed image and gt
    for i in range(len(imagesUOrigonal)):
        image = imagesUOrigonal[i]
        saveName = args.output_data_dir + "/images/"+ str(j) + "_" + args.data_extension
        imageio.imsave(saveName, image)
        # focus blur default for 0 pixels
        saliencyBMask = imagesESaliency[i]
        (T, saliency) = cv2.threshold(saliencyBMask, .1, 1, cv2.THRESH_BINARY)
        imagegray = skimage.color.rgb2gray(image)*255
        saliency[saliency == 0] = 128  # focus
        saliency[saliency == 1] = 192  # underexposure
        saliency[imagegray < 45] = 192
        saliency[imagegray > 225] = 255
        saveName = args.output_data_dir + "/gt/"+str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliency)
        j += 1

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS MURI CREATE BLUR TESTING DATASET FROM FETCH')
    # directory data location
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_data_dir', type=str, default=None)
    parser.add_argument('--salinecy_exposed_data_dir', type=str, default=None)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str, default=".png")
    args = parser.parse_args()
    create_dataset_for_testing(args)


