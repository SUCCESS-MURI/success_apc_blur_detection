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

# for my images
gamma = 1.5

#create out of focus blur for 3 channel images
def apply_out_of_focus_blur(image, kernelsize):
    # create disk kernal
    kernal = disk(kernelsize)
    kernal = kernal / kernal.sum()
    # now convolve
    # https://www.askpython.com/python-modules/opencv-filter2d
    image_blurred = cv2.filter2D(image, -1, kernal)
    image_blurred[image_blurred < 0] = 0
    return image_blurred

def random_focus_blur_kernel(dmin=10, dmax=50):
    random_kernal_size = random.randint(dmin, dmax)
    return random_kernal_size

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
        imagegray = skimage.color.rgb2gray(image) * 255
        saliency[imagegray < 30] = 192
        saliency[imagegray > 245] = 255
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
        imagegray = skimage.color.rgb2gray(image) * 255
        saliencyBMask[imagegray < 30] = 192
        saliencyBMask[imagegray > 245] = 255
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
        imagegray = skimage.color.rgb2gray(image) * 255
        saliencyBMask[imagegray < 30] = 192
        saliencyBMask[imagegray > 245] = 255
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
        saliency[imagegray < 30] = 192
        saliency[imagegray > 245] = 255
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
        saliency[imagegray < 30] = 192
        saliency[imagegray > 245] = 255
        saveName = args.output_data_dir + "/gt/"+str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliency)
        j += 1

# create testing dataset for chuk with brightness and darkness images
def create_dataset_for_testing_focus(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_normal_list = sorted(tl.files.load_file_list(path=args.data_dir +'/normal', regx='/*.(png|PNG)',
                                                        printable=False))
    # images_focus_list = sorted(tl.files.load_file_list(path=args.data_dir +'/focus', regx='/*.(png|PNG)',
    #                                                    printable=False))
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
    #imagesFOrigonal = read_all_imgs(images_focus_list, path=args.data_dir +'/focus/', n_threads=100, mode='RGB')
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
        imagegray = skimage.color.rgb2gray(image) * 255
        saliency[imagegray < np.mean(imagegray)/3] = 192
        saliency[imagegray > 245] = 255
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
        imagegray = skimage.color.rgb2gray(image) * 255
        saliencyBMask[imagegray < np.mean(imagegray)/3] = 192
        saliencyBMask[imagegray > 245] = 255
        saveName = args.output_data_dir + "/gt/"+str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliencyBMask)
        j += 1

    # save focus images
    # 1st need to get the focus image
    # 2nd need to save focus image and gt
    for i in range(len(imagesNOrigonal)):
        image = imagesNOrigonal[i]
        image_focus = 255.0 * np.power((image * 1.) / 255.0, gamma)
        focus_kernal = random_focus_blur_kernel()
        image_focus = apply_out_of_focus_blur(image_focus, focus_kernal)
        image = np.round(np.power((image_focus * 1.) / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
        saveName = args.output_data_dir + "/images/" + str(j) + "_" + args.data_extension
        imageio.imsave(saveName, image)
        # motion blur
        saliencyBMask = np.ones(final_shape) * 128
        imagegray = skimage.color.rgb2gray(image) * 255
        saliencyBMask[imagegray < 10] = 192
        saliencyBMask[imagegray > 245] = 255
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
        saliency[imagegray < np.mean(imagegray)/3] = 192
        saliency[imagegray > 245] = 255
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
        saliency[imagegray < 30] = 192
        saliency[imagegray > 245] = 255
        saveName = args.output_data_dir + "/gt/"+str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliency)
        j += 1

# create testing dataset for chuk with brightness and darkness images
# focus images are base again
def create_dataset_for_testing_motion(args):
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
    images_sailency_motion_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency_motion',
                                                                 regx='/*.(png|PNG)',printable=False))
    images_sailency_exposed_list = sorted(tl.files.load_file_list(path=args.data_dir + '/salinecy_exposed',
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
    imagesMSaliency = read_all_imgs(images_sailency_motion_list, path=args.data_dir + '/saliency_motion/',
                                   n_threads=100, mode='GRAY')
    imagesESaliency = read_all_imgs(images_sailency_exposed_list, path=args.data_dir + '/salinecy_exposed/',
                                    n_threads=100,
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
        imagegray = skimage.color.rgb2gray(image) * 255
        saliency[imagegray < np.mean(imagegray)/3] = 192
        saliency[imagegray > 245] = 255
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
        saliencyBMask = imagesMSaliency[i]
        (T, saliency) = cv2.threshold(saliencyBMask, .001, 1, cv2.THRESH_BINARY)
        saliency[saliency == 0] = 128  # focus
        saliency[saliency == 1] = 64  # motion
        imagegray = skimage.color.rgb2gray(image) * 255
        saliency[imagegray < np.mean(imagegray) / 3] = 192
        saliency[imagegray > 245] = 255
        # saliencyBMask = np.ones(final_shape)*64
        saveName = args.output_data_dir + "/gt/"+str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliency)
        j += 1

    # save focus images
    # 1st need to get the focus image
    # 2nd need to save focus image and gt
    for i in range(len(imagesFOrigonal)):
        # image = imagesNOrigonal[i]
        # image_focus = 255.0 * np.power((image * 1.) / 255.0, gamma)
        # focus_kernal = random_focus_blur_kernel()
        # image_focus = apply_out_of_focus_blur(image_focus, focus_kernal)
        # image = np.round(np.power((image_focus * 1.) / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
        image = imagesFOrigonal[i]
        saveName = args.output_data_dir + "/images/" + str(j) + "_" + args.data_extension
        imageio.imsave(saveName, image)
        # focus blur
        saliencyBMask = np.ones(final_shape) * 128
        imagegray = skimage.color.rgb2gray(image) * 255
        saliencyBMask[imagegray < 10] = 192
        saliencyBMask[imagegray > 245] = 255
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
        saliency[imagegray < np.mean(imagegray)/3] = 192
        saliency[imagegray > 245] = 255
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
        saliency[imagegray < 30] = 192
        saliency[imagegray > 245] = 255
        saveName = args.output_data_dir + "/gt/"+str(j) + "_" + args.data_extension
        cv2.imwrite(saveName, saliency)
        j += 1

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS MURI CREATE BLUR TESTING DATASET FROM FETCH')
    # directory data location
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_data_dir', type=str, default=None)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str, default=".png")
    args = parser.parse_args()
    #create_dataset_for_testing(args)
    #create_dataset_for_testing_focus(args)
    create_dataset_for_testing_motion(args)


