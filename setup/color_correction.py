# Author mhatfalv
# Create the blur dataset
# import display() to show final image
import argparse
import copy
import glob
#import os
import random
#import sys

import cv2
import numpy as np
# create motion blur for image
#from wand.image import Image
#sys.path.insert(0,os.environ["SUCCESS_APC"])
from PIL import Image as im, ImageEnhance
import tensorlayer as tl
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.exposure import match_histograms

# alpha = 1.0
# alpha_max = 500
# beta = 0
# beta_max = 200
# gamma = 1.0
# gamma_max = 300
#
# def basicLinearTransform():
#     res = cv2.convertScaleAbs(img_original, alpha=alpha, beta=beta)
#     img_corrected = cv2.hconcat([img_original, res])
#     cv2.imshow("Brightness and contrast adjustments", img_corrected)
#
# def gammaCorrection():
#     ## [changing-contrast-brightness-gamma-correction]
#     lookUpTable = np.empty((1,256), np.uint8)
#     for i in range(256):
#         lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
#
#     res = cv2.LUT(img_original, lookUpTable)
#     ## [changing-contrast-brightness-gamma-correction]
#
#     img_gamma_corrected = cv2.hconcat([img_original, res])
#     cv2.imshow("Gamma correction", img_gamma_corrected)
#
# def on_linear_transform_alpha_trackbar(val):
#     global alpha
#     alpha = val / 100
#     basicLinearTransform()
#
# def on_linear_transform_beta_trackbar(val):
#     global beta
#     beta = val - 100
#     basicLinearTransform()
#
# def on_gamma_correction_trackbar(val):
#     global gamma
#     gamma = val / 100
#     gammaCorrection()


# this takes all the images and then overlays the image with random masked images that are blurred
def color_correct_image(args):
    tl.files.exists_or_mkdir(args.output_data_dir)
    #tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    #i = 0
    gamma_init = 1.0
    list = glob.glob(args.data_dir + '/*'+ args.data_extension)
    list.sort(reverse=True)
    for imageFileName in list:
        # global img_original
        image = cv2.imread(imageFileName)
        # img_original = image
        #
        # img_corrected = np.empty((img_original.shape[0], img_original.shape[1] * 2, img_original.shape[2]),
        #                          img_original.dtype)
        # img_gamma_corrected = np.empty((img_original.shape[0], img_original.shape[1] * 2, img_original.shape[2]),
        #                                img_original.dtype)
        #
        # img_corrected = cv2.hconcat([img_original, img_original])
        # img_gamma_corrected = cv2.hconcat([img_original, img_original])
        #
        # cv2.namedWindow('Brightness and contrast adjustments')
        # cv2.namedWindow('Gamma correction')
        #
        # alpha_init = int(alpha * 100)
        # cv2.createTrackbar('Alpha gain (contrast)', 'Brightness and contrast adjustments', alpha_init, alpha_max,
        #                   on_linear_transform_alpha_trackbar)
        # beta_init = beta + 100
        # cv2.createTrackbar('Beta bias (brightness)', 'Brightness and contrast adjustments', beta_init, beta_max,
        #                   on_linear_transform_beta_trackbar)
        # gamma_init = int(gamma * 100)
        # cv2.createTrackbar('Gamma correction', 'Gamma correction', gamma_init, gamma_max, on_gamma_correction_trackbar)
        #
        # on_linear_transform_alpha_trackbar(alpha_init)
        # on_gamma_correction_trackbar(gamma_init)
        #
        # cv2.waitKey(0)

        #image = cv2.imread('cathedral.jpg')
        # src = cv2.imread(imageFileName)
        # if i == 0:
        #     ref = src
        #     i = 1
        # # determine if we are performing multichannel histogram matching
        # # and then perform histogram matching itself
        # print("[INFO] performing histogram matching...")
        # multi = True if src.shape[-1] > 1 else False
        # matched = match_histograms(src, ref, multichannel=multi)
        # cv2.imshow("Source", src)
        # cv2.imshow("Reference", ref)
        # cv2.imshow("Matched", matched)
        # cv2.waitKey(0)
        #cv2.imwrite(args.output_data_dir+imageFileName.split('/')[-1],matched)

if __name__ == "__main__":
 # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS CREATE BLUR DATASET')
    # directory data location
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_data_dir', type=str)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str,default=".png")
    parser.add_argument('--is_testing', default=False, action='store_true')
    args = parser.parse_args()
    color_correct_image(args)
