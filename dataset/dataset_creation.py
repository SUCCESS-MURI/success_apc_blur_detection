# Author mhatfalv
# Create the blur dataset
# import display() to show final image
import argparse
import copy
import glob
#import os
import random
#import sys
import imageio
import pyblur
import cv2
import numpy as np
# create motion blur for image
#from wand.image import Image
#sys.path.insert(0,os.environ["SUCCESS_APC"])
from pyblur import LinearMotionBlur_random

import saliency_detection
from PIL import Image as im, ImageEnhance
import tensorlayer as tl

# https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
#size - in pixels, size of motion blur
#angel - in degrees, direction of motion blur
#from dataset.resize_dataset import resize_dataset
from utils import read_all_imgs

# gamma value
gamma = 2.2

# def apply_motion_blur(image, size, angle):
#     k = np.zeros((size, size), dtype=np.float32)
#     k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
#     k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
#     k = k * (1.0 / np.sum(k))
#     return cv2.filter2D(image, -1, k)

# from https://github.com/Imalne/Defocus-and-Motion-Blur-Detection-with-Deep-Contextual-Features
# /blob/a368a3e0a8869011ec167bb1f8eb82ceff091e0c/DataCreator/Blend.py#L14

######## motion blur #########
def apply_motion_blur(image, size, angle):
    Motion = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = np.diag(np.ones(size))
    kernel = cv2.warpAffine(kernel, Motion, (size, size))
    kernel = kernel / size
    blurred = cv2.filter2D(image, -1, kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return blurred

def normalvariate_random_int(mean, variance, dmin, dmax):
    r = dmax + 1
    while r < dmin or r > dmax:
        r = int(random.normalvariate(mean, variance))
    return r

def uniform_random_int(dmin, dmax):
    r = random.randint(dmin,dmax)
    return r

def random_motion_blur_kernel(mean=50, variance=15, dmin=10, dmax=100):
    random_degree = normalvariate_random_int(mean, variance, dmin, dmax)
    random_angle = uniform_random_int(-180, 180)
    return random_degree,random_angle

##########################################

def random_darkness_value(amin=0.01,amax=0.6,bmin=-200,bmax=0):
    alpha = random.uniform(amin,amax)
    beta = random.uniform(bmin,bmax)
    return alpha,beta

def random_brightness_value(amin=1.5,amax=2.5,bmin=100,bmax=200):
    alpha = random.uniform(amin,amax)
    beta = random.uniform(bmin,bmax)
    return alpha,beta

# create out of focus blur for 3 channel images
def create_out_of_focus_blur(image,kernelsize):
    image_blurred = cv2.blur(image,(kernelsize,kernelsize))
    return image_blurred

def create_brightness_and_darkness_blur(image, alpha, beta):
    new_img = image * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img

# https://github.com/Imalne/Defocus-and-Motion-Blur-Detection-with-Deep-Contextual-Features/blob/master/DataCreator/Blend.py#L5
def alpha_blending(defocus,motion,alpha):
    f_defocus = defocus.astype("float32")
    f_motion = motion.astype("float32")
    f_blended = f_defocus*(1-alpha) + f_motion * alpha
    return f_blended#.astype("uint8")

# from https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.
    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    return img,y1o,y2o,x1o,x2o

def create_chuk_dataset_for_training(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir + '/images', regx='/*.(jpg|JPG)', printable=False))
    images_gt_list = sorted(tl.files.load_file_list(path=args.data_dir+'/gt', regx='/*.(png|PNG)', printable=False))
    saliency_images_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir + '/images', regx='/*.(jpg|JPG)',
                                                          printable=False))
    saliency_mask_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir + '/gt', regx='/*.(png|PNG)',
                                                          printable=False))
    # these images have out of focus blur already
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir+'/images/', n_threads=100, mode='RGB')
    gtOrigonal = read_all_imgs(images_gt_list, path=args.data_dir+'/gt/', n_threads=100, mode='RGB2GRAY2')
    saliencyImages = read_all_imgs(saliency_images_list, path=args.salinecy_data_dir +'/images/', n_threads=100,
                                   mode='RGB')
    saliencyMask = read_all_imgs(saliency_mask_list, path=args.salinecy_data_dir+'/gt/', n_threads=100,
                                 mode='RGB2GRAY2')

    for i in range(len(imagesOrigonal)):
        if imagesOrigonal[i].shape[0] > imagesOrigonal[i].shape[1]:
            imagesOrigonal[i] = cv2.rotate(imagesOrigonal[i],0)
            gtOrigonal[i] = cv2.rotate(gtOrigonal[i], 0)[:,:,np.newaxis]
        y1, y2 = max(0, int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2)), \
                 min(imagesOrigonal[i].shape[0]+int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2),
                     final_shape[0])
        x1, x2 = max(0, int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2)), \
                 min(imagesOrigonal[i].shape[1]+int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2),
                     final_shape[1])
        y1o, y2o = 0, min(imagesOrigonal[i].shape[0], final_shape[0])
        x1o, x2o = 0, min(imagesOrigonal[i].shape[1], final_shape[1])

        new_image = np.zeros((final_shape[0], final_shape[1], 3))
        new_image[y1:y2, x1:x2] = imagesOrigonal[i][y1o:y2o,x1o:x2o]
        # darkness blur deafult for 0 pixels
        gtMask =  np.ones((final_shape[0], final_shape[1], 1))
        gtMask[y1:y2, x1:x2] = gtOrigonal[i][y1o:y2o,x1o:x2o]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        images_list[i] = images_list[i].split('.')[0]
        new_mask = np.zeros(gtMask.shape)
        new_mask[gtMask == 0] = 0
        new_mask[gtMask == 1] = 3
        new_mask[gtMask == 255] = 2
        gtOrigonal[i] = new_mask

    for i in range(len(saliencyImages)):
        if saliencyImages[i].shape[0] > saliencyImages[i].shape[1]:
            saliencyImages[i] = cv2.rotate(saliencyImages[i], 0)
            saliencyMask[i] = cv2.rotate(saliencyMask[i], 0)[:, :, np.newaxis]
        y1, y2 = max(0, int((final_shape[0] + 1 - saliencyImages[i].shape[0]) / 2)), \
                 min(saliencyImages[i].shape[0] + int((final_shape[0] + 1 - saliencyImages[i].shape[0]) / 2),
                     final_shape[0])
        x1, x2 = max(0, int((final_shape[1] + 1 - saliencyImages[i].shape[1]) / 2)), \
                 min(saliencyImages[i].shape[1] + int((final_shape[1] + 1 - saliencyImages[i].shape[1]) / 2),
                     final_shape[1])
        y1o, y2o = 0, min(saliencyImages[i].shape[0], final_shape[0])
        x1o, x2o = 0, min(saliencyImages[i].shape[1], final_shape[1])

        new_image = np.zeros((final_shape[0], final_shape[1], 3))
        new_image[y1:y2, x1:x2] = saliencyImages[i][y1o:y2o, x1o:x2o]
        new_mask = np.zeros((final_shape[0], final_shape[1], 1))
        new_mask[y1:y2, x1:x2] = saliencyMask[i][y1o:y2o, x1o:x2o]
        saliencyImages[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        (T, saliency) = cv2.threshold(new_mask, 1, 1, cv2.THRESH_BINARY)
        saliencyMask[i] = saliency

    # now we will go through all of the images and make the dataset
    indexs = np.arange(start=0, stop=len(imagesOrigonal), step=1)
    np.random.shuffle(indexs)
    idxs = np.arange(start=0, stop=3, step=1)
    saliencyIdx = np.arange(start=0, stop=len(saliencyImages), step=1)
    # this already makes a huge dataset
    for i in range(len(imagesOrigonal)):
        baseImageName = images_list[i]
        np.random.shuffle(idxs)
        for count in range(5):
            final_masked_blurred_image = copy.deepcopy(imagesOrigonal[i])
            nMask = copy.deepcopy(gtOrigonal[i])
            for j in idxs:
                # motion blur
                if j == 0:
                    overlayImageIdx = np.random.choice(saliencyIdx, 1, replace=False)[0]
                    kernal,angle = random_motion_blur_kernel()
                    motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(saliencyImages[overlayImageIdx]), kernal, angle)
                    # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
                    # cv2.waitKey(0)
                    blur_mask = apply_motion_blur(copy.deepcopy(saliencyMask[overlayImageIdx]) * 255, kernal, angle)
                    motion_blur_mask = copy.deepcopy(blur_mask)
                    motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                    alpha = blur_mask / 255.
                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                            motion_blurred_overlay_img, placement[1],placement[0],alpha)
                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    nMask[max(0, placement[0]) + np.argwhere(motion_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,0],
                      max(0, placement[1]) + np.argwhere(motion_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,1]] = 1
                    # cv2.imshow('BrightnessImage', nMask)
                    # cv2.waitKey(0)
                # darkness blur
                elif j == 1:
                    # create darkness blur
                    overlayImageIdx = np.random.choice(saliencyIdx, 1, replace=False)[0]
                    alpha,beta = random_darkness_value()
                    dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[overlayImageIdx]),
                                                                               alpha, beta)
                    # cv2.imshow('BrightnessImage', dark_blur)
                    # cv2.waitKey(0)
                    dark_blur_mask = copy.deepcopy(saliencyMask[overlayImageIdx])
                    dark_blur_mask[dark_blur_mask > 0] = 255
                    alpha_mask = (copy.deepcopy(saliencyMask[overlayImageIdx]) * 255.0) / 255.
                    # cv2.imshow('BrightnessImage_mask', dark_blur_masked)
                    # cv2.waitKey(0)
                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                             np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                    dark_blurred_overlay_img, placement[1],placement[0],alpha_mask)
                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # add darkness blur to mask image
                    # indicator for dark blur
                    nMask[max(0, placement[0]) + np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,0],
                      max(0, placement[1]) + np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,1]] = 3
                # brightness blur
                elif j == 2:
                    # create brightness blur
                    overlayImageIdx = np.random.choice(saliencyIdx, 1, replace=False)[0]
                    alpha, beta = random_brightness_value()
                    bright_blurred_overlay_img = create_brightness_and_darkness_blur(
                        copy.deepcopy(saliencyImages[overlayImageIdx]), alpha, beta)
                    # cv2.imshow('BrightnessImage', dark_blur)
                    # cv2.waitKey(0)
                    bright_blur_mask = copy.deepcopy(saliencyMask[overlayImageIdx])
                    bright_blur_mask[bright_blur_mask > 0] = 255
                    alpha_mask = (copy.deepcopy(saliencyMask[overlayImageIdx]) * 255.0) / 255.
                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                             np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                bright_blurred_overlay_img, placement[1],placement[0], alpha_mask)
                    # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # add brightness blur to mask image
                    # indicator for brightness
                    nMask[max(0, placement[0]) + np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,0],
                      max(0, placement[1]) + np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,1]] = 4
            # save final image
            saveName = args.output_data_dir + "/images/" + baseImageName + '_' + str(count) + args.data_extension
            final_masked_blurred_image = np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                                                  / 255.0, (1.0 / gamma)) * 255.0
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_' + str(count) + args.data_extension
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveNameGt, nMask)

def create_chuk_dataset_for_testing_and_validation(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='/*.(png|PNG)', printable=False))
    saliency_images_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir, regx='/*.(png|PNG)',
                                                          printable=False))
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir, n_threads=100, mode='RGB')
    SaliencyImages = read_all_imgs(saliency_images_list, path=args.salinecy_data_dir, n_threads=100, mode='RGB2GRAY2')
    maskSaliency = []
    for i in range(len(imagesOrigonal)):
        new_image = np.zeros((final_shape[0], final_shape[1], 3))
        new_mask = np.zeros((final_shape[0], final_shape[1], 1))
        padd_h = int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2)
        padd_w = int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2)
        new_image[padd_h:imagesOrigonal[i].shape[0] + padd_h, padd_w:imagesOrigonal[i].shape[1] + padd_w] = \
            imagesOrigonal[i]
        new_mask[padd_h:imagesOrigonal[i].shape[0] + padd_h, padd_w:imagesOrigonal[i].shape[1] + padd_w] = \
            SaliencyImages[i]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255, gamma)
        (T, saliency) = cv2.threshold(new_mask, 1, 1, cv2.THRESH_BINARY)
        SaliencyImages[i] = saliency
        maskSaliency.append(saliency[:, :, np.newaxis] * imagesOrigonal[i])
        images_list[i] = images_list[i].split('.')[0]

    # now we will go through all of the images and make the dataset
    indexs = np.arange(start=0, stop=len(imagesOrigonal), step=1)
    # TODO need to figure this out (might just want to do the whole image)
    for i in range(len(imagesOrigonal)):
        idx = indexs[i]
        baseImageName = images_list[idx]
        for s in kernal:
            for a in angles:
                # motion blur
                motion_blurred_overlay_img = apply_motion_blur(np.copy(maskSaliency[idx]), s, a)
                blur_mask = apply_motion_blur(np.copy(SaliencyImages[idx]) * 255, s, a)
                motion_blur_mask = np.copy(blur_mask)
                motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                alpha = blur_mask / 255.0
                final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), motion_blurred_overlay_img,
                                                            alpha[:, :, np.newaxis])
                # now save the image
                saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
                saveName = args.output_data_dir + "/gt/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                # cv2.waitKey(0)
                nMask = np.zeros(final_shape)
                nMask[motion_blur_mask == 255] = 64
                cv2.imwrite(saveName, nMask)
                # cv2.imshow('BrightnessImage', nMask)
                # cv2.waitKey(0)
        ### out of focus blur
        for k in kernal:
            focus_blur_overlay_img = create_out_of_focus_blur(np.copy(maskSaliency[idx]), k)
            blur_mask = create_out_of_focus_blur(np.copy(SaliencyImages[idx]) * 255.0, k)
            focus_blur_mask = np.copy(blur_mask)
            focus_blur_mask[np.round(blur_mask, 4) > 0] = 255
            alpha = blur_mask / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), focus_blur_overlay_img,
                                                        alpha[:, :, np.newaxis])
            saveName = args.output_data_dir + "/images/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            # add focus identification to mask
            nMask = np.zeros(final_shape)
            nMask[focus_blur_mask == 255] = 128
            cv2.imwrite(saveName, nMask)
        # next we go to darkness blur
        for a in alpha_darkness:
            # create darkness blur
            dark_blur = create_brightness_and_darkness_blur(np.copy(maskSaliency[idx]), a, 0)
            dark_blur_mask = np.copy(SaliencyImages[idx])
            dark_blur_mask[dark_blur_mask > 0] = 255
            alpha = (np.copy(SaliencyImages[idx]) * 255.0) / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), dark_blur,
                                                        alpha[:, :, np.newaxis])
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            # add darkness blur to mask image
            nMask = np.zeros(final_shape)
            nMask[dark_blur_mask == 255] = 192
            cv2.imwrite(saveName, nMask)
        # next we go to brightness blur
        for a in alpha_brightness:
            # create brightness blur
            bright_blur = create_brightness_and_darkness_blur(np.copy(maskSaliency[idx]), a, 0)
            bright_blur_mask = np.copy(SaliencyImages[idx])
            bright_blur_mask[bright_blur_mask > 0] = 255
            alpha = (np.copy(SaliencyImages[idx]) * 255.0) / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), bright_blur,
                                                        alpha[:, :, np.newaxis])
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            # add brightness blur to mask image
            nMask = np.zeros(final_shape)
            nMask[bright_blur_mask == 255] = 255
            cv2.imwrite(saveName, nMask)
    # save control / no blur images
    for n in range(2):
        for i in range(len(imagesOrigonal)):
            # always remember gamma correction
            baseImage = np.power((imagesOrigonal[i] * 1.) / 255, (1 / gamma)) * 255.0
            baseImageName = images_list[i]
            saveName = args.output_data_dir + "/images/" + baseImageName + "_no_blur_" + str(
                n + 1) + args.data_extension
            cv2.imwrite(saveName, baseImage)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_no_blur_" + str(
                n + 1) + args.data_extension
            nMask = np.zeros(final_shape)
            cv2.imwrite(saveName, nMask)

def create_ssc_dataset_for_training(args):
    # angles range for dataset
    angles = np.arange(start=0, stop=190, step=30)
    kernal = np.arange(start=3, stop=26, step=4)
    # alpha
    alpha_darkness = np.arange(start=0.1, stop=0.6, step=0.1)
    alpha_brightness = np.arange(start=2.0, stop=2.5, step=0.1)
    # beta
    # beta = np.arange(start=0,stop=1,step=1) # 110 step 10
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='/*.(png|PNG)', printable=False))
    saliency_images_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir, regx='/*.(png|PNG)',
                                                          printable=False))
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir, n_threads=100, mode='RGB')
    SaliencyImages = read_all_imgs(saliency_images_list, path=args.salinecy_data_dir, n_threads=100, mode='RGB2GRAY2')
    maskSaliency = []
    for i in range(len(imagesOrigonal)):
        new_image = np.zeros((final_shape[0], final_shape[1], 3))
        new_mask = np.zeros((final_shape[0], final_shape[1], 1))
        padd_h = int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2)
        padd_w = int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2)
        new_image[padd_h:imagesOrigonal[i].shape[0] + padd_h, padd_w:imagesOrigonal[i].shape[1] + padd_w] = \
        imagesOrigonal[i]
        new_mask[padd_h:imagesOrigonal[i].shape[0] + padd_h, padd_w:imagesOrigonal[i].shape[1] + padd_w] = \
        SaliencyImages[i]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255, gamma)
        (T, saliency) = cv2.threshold(new_mask, 1, 1, cv2.THRESH_BINARY)
        SaliencyImages[i] = saliency
        maskSaliency.append(saliency[:, :, np.newaxis] * imagesOrigonal[i])
        images_list[i] = images_list[i].split('.')[0]

    # now we will go through all of the images and make the dataset
    indexs = np.arange(start=0, stop=len(imagesOrigonal), step=1)
    np.random.shuffle(indexs)
    idxs = np.arange(start=0, stop=4, step=1)
    # this already makes a huge dataset
    for i in range(1):
        tempIdxs = indexs[indexs != i]
        baseImage = imagesOrigonal[i]
        baseImageName = images_list[i]
        for s in kernal:
            for a in angles:
                for k in kernal:
                    for al_d in alpha_darkness:
                        for al_b in alpha_brightness:
                            # need to study this more if training will be affected
                            final_masked_blurred_image = np.copy(baseImage)
                            nMask = np.zeros(final_shape)
                            np.random.shuffle(idxs)
                            for j in idxs:
                                # motion blur
                                if j == 0:
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    motion_blurred_overlay_img = apply_motion_blur(
                                        np.copy(maskSaliency[overlayImageIdx]), s, a)
                                    # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
                                    # cv2.waitKey(0)
                                    blur_mask = apply_motion_blur(np.copy(SaliencyImages[overlayImageIdx]) * 255, s, a)
                                    motion_blur_mask = np.copy(blur_mask)
                                    motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                                    alpha = blur_mask / 255.
                                    # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
                                    # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
                                    # motion_blurred_imgPIL,alpha_PIL = convert_whiten_image(motion_blurred_overlay_img,
                                    #                                         im.fromarray(alpha).convert('RGBA'))
                                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(
                                        final_masked_blurred_image, motion_blurred_overlay_img, placement[1],
                                        placement[0],
                                        alpha)
                                    # # https://note.nkmk.me/en/python-pillow-paste/
                                    # final_masked_blurred_image.paste(motion_blurred_imgPIL, placement, mask=alpha_PIL)
                                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    nMask[
                                        max(0, placement[0]) + np.argwhere(motion_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,
                                                               0],
                                        max(0, placement[1]) + np.argwhere(motion_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,
                                                               1]] = 64
                                    # cv2.imshow('BrightnessImage', nMask)
                                    # cv2.waitKey(0)
                                    # focus blur
                                elif j == 1:
                                    # now do out of focus blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    focus_blurred_overlay_img = create_out_of_focus_blur(
                                        np.copy(maskSaliency[overlayImageIdx]), k)
                                    blur_mask = create_out_of_focus_blur(np.copy(SaliencyImages[overlayImageIdx]) * 255,
                                                                         k)
                                    focus_blur_mask = np.copy(blur_mask)
                                    focus_blur_mask[np.round(blur_mask, 4) > 0] = 255
                                    alpha = blur_mask / 255.
                                    # focus_blurred_imgPIL, alpha_PIL = convert_whiten_image(focus_blurred_overlay_img,
                                    #                                        im.fromarray(alpha).convert('RGBA'))
                                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(
                                        final_masked_blurred_image, focus_blurred_overlay_img, placement[1],
                                        placement[0],
                                        alpha)
                                    # final_masked_blurred_image.paste(focus_blurred_imgPIL, placement,
                                    #                                  mask=alpha_PIL)
                                    # cv2.imshow('FocusImage',final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add focus identification to mask
                                    # idx = np.where(np.array(alpha_PIL)[:, :, 3] == 255)
                                    nMask[
                                        max(0, placement[0]) + np.argwhere(focus_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,
                                                               0],
                                        max(0, placement[1]) + np.argwhere(focus_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,
                                                               1]] = 128
                                # darkness blur
                                elif j == 2:
                                    # create darkness blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    dark_blurred_overlay_img = create_brightness_and_darkness_blur(
                                        np.copy(maskSaliency[overlayImageIdx]), al_d, 0)
                                    # cv2.imshow('BrightnessImage', dark_blur)
                                    # cv2.waitKey(0)
                                    dark_blur_mask = np.copy(SaliencyImages[overlayImageIdx])
                                    dark_blur_mask[dark_blur_mask > 0] = 255
                                    alpha = (np.copy(SaliencyImages[overlayImageIdx]) * 255.0) / 255.
                                    # cv2.imshow('BrightnessImage_mask', dark_blur_masked)
                                    # cv2.waitKey(0)
                                    # darkness = True
                                    # dark_blurred_imgPIL,alpha_PIL = convert_whiten_image(dark_blurred_overlay_img,
                                    #                                             im.fromarray(alpha).convert('RGBA'))
                                    # # cv2.imshow('BrightnessImage2', np.array(dark_blurred_imgPIL)[:,:,0:3])
                                    # # cv2.waitKey(0)
                                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(
                                        final_masked_blurred_image, dark_blurred_overlay_img, placement[1],
                                        placement[0],
                                        alpha)
                                    # final_masked_blurred_image.paste(dark_blurred_imgPIL, placement,
                                    #                                 mask=alpha_PIL)

                                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add brightness and/or darkness blur to mask image
                                    # indicator for dark blur
                                    nMask[max(0, placement[0]) + np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,
                                                                 0],
                                          max(0, placement[1]) + np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,
                                                                 1]] = 192
                                # brightness blur
                                elif j == 3:
                                    # create brightness blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    bright_blurred_overlay_img = create_brightness_and_darkness_blur(
                                        np.copy(maskSaliency[overlayImageIdx]), al_b, 0)
                                    # cv2.imshow('BrightnessImage', dark_blur)
                                    # cv2.waitKey(0)
                                    bright_blur_mask = np.copy(SaliencyImages[overlayImageIdx])
                                    bright_blur_mask[bright_blur_mask > 0] = 255
                                    alpha = (np.copy(SaliencyImages[overlayImageIdx]) * 255.0) / 255.
                                    # bright_blurred_imgPIL, alpha_PIL = convert_whiten_image(bright_blurred_overlay_img,
                                    #                                         im.fromarray(alpha).convert('RGBA'))
                                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(
                                        final_masked_blurred_image, bright_blurred_overlay_img, placement[1],
                                        placement[0], alpha)
                                    # final_masked_blurred_image.paste(bright_blurred_imgPIL, placement,mask=alpha_PIL)
                                    # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add brightness and/or darkness blur to mask image
                                    # idx = np.where(np.array(alpha_PIL)[:, :, 3] == 255)
                                    # indicator for brightness
                                    nMask[
                                        max(0, placement[0]) + np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,
                                                               0],
                                        max(0, placement[1]) + np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:,
                                                               1]] = 255
                            # save final image
                            saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur" \
                                       + "_a_" + str(a) + "_s_" + str(s) + "_focus_blur_k_" + str(k) \
                                       + "_darkness_blur_al_" + str(al_d) + "_brightness_blur_al_" + str(al_b) + \
                                       args.data_extension
                            final_masked_blurred_image = np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                                                  / 255, (1 / gamma)) * 255.0
                            cv2.imwrite(saveName, final_masked_blurred_image)
                            # create and save ground truth mask
                            saveNameGt = args.output_data_dir + "/gt/" + baseImageName + "_motion_blur" \
                                         + "_a_" + str(a) + "_s_" + str(s) + "_focus_blur_k_" + str(k) \
                                         + "_darkness_blur_al_" + str(al_d) + "_brightness_blur_al_" + str(al_b) + \
                                         args.data_extension
                            cv2.imwrite(saveNameGt, nMask)
    # random chose the next 5 index
    # TODO need to figure this out (might just want to do the whole image)
    for i in range(len(imagesOrigonal)):
        idx = indexs[i]
        baseImageName = images_list[idx]
        for s in kernal:
            for a in angles:
                # motion blur
                motion_blurred_overlay_img = apply_motion_blur(np.copy(maskSaliency[idx]), s, a)
                blur_mask = apply_motion_blur(np.copy(SaliencyImages[idx]) * 255, s, a)
                motion_blur_mask = np.copy(blur_mask)
                motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                alpha = blur_mask / 255.0
                final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), motion_blurred_overlay_img,
                                                            alpha[:, :, np.newaxis])
                # now save the image
                saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
                saveName = args.output_data_dir + "/gt/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                # cv2.waitKey(0)
                nMask = np.zeros(final_shape)
                nMask[motion_blur_mask == 255] = 64
                cv2.imwrite(saveName, nMask)
                # cv2.imshow('BrightnessImage', nMask)
                # cv2.waitKey(0)
        ### out of focus blur
        for k in kernal:
            focus_blur_overlay_img = create_out_of_focus_blur(np.copy(maskSaliency[idx]), k)
            blur_mask = create_out_of_focus_blur(np.copy(SaliencyImages[idx]) * 255.0, k)
            focus_blur_mask = np.copy(blur_mask)
            focus_blur_mask[np.round(blur_mask, 4) > 0] = 255
            alpha = blur_mask / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), focus_blur_overlay_img,
                                                        alpha[:, :, np.newaxis])
            saveName = args.output_data_dir + "/images/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            # add focus identification to mask
            nMask = np.zeros(final_shape)
            nMask[focus_blur_mask == 255] = 128
            cv2.imwrite(saveName, nMask)
        # next we go to darkness blur
        for a in alpha_darkness:
            # create darkness blur
            dark_blur = create_brightness_and_darkness_blur(np.copy(maskSaliency[idx]), a, 0)
            dark_blur_mask = np.copy(SaliencyImages[idx])
            dark_blur_mask[dark_blur_mask > 0] = 255
            alpha = (np.copy(SaliencyImages[idx]) * 255.0) / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), dark_blur,
                                                        alpha[:, :, np.newaxis])
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            # add darkness blur to mask image
            nMask = np.zeros(final_shape)
            nMask[dark_blur_mask == 255] = 192
            cv2.imwrite(saveName, nMask)
        # next we go to brightness blur
        for a in alpha_brightness:
            # create brightness blur
            bright_blur = create_brightness_and_darkness_blur(np.copy(maskSaliency[idx]), a, 0)
            bright_blur_mask = np.copy(SaliencyImages[idx])
            bright_blur_mask[bright_blur_mask > 0] = 255
            alpha = (np.copy(SaliencyImages[idx]) * 255.0) / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), bright_blur,
                                                        alpha[:, :, np.newaxis])
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            # add brightness blur to mask image
            nMask = np.zeros(final_shape)
            nMask[bright_blur_mask == 255] = 255
            cv2.imwrite(saveName, nMask)
    # save control / no blur images
    for n in range(500):
        for i in range(len(imagesOrigonal)):
            # always remember gamma correction
            baseImage = np.power((imagesOrigonal[i] * 1.) / 255, (1 / gamma)) * 255.0
            baseImageName = images_list[i]
            saveName = args.output_data_dir + "/images/" + baseImageName + "_no_blur_" + str(
                n + 1) + args.data_extension
            cv2.imwrite(saveName, baseImage)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_no_blur_" + str(
                n + 1) + args.data_extension
            nMask = np.zeros(final_shape)
            cv2.imwrite(saveName, nMask)

def create_ssc_dataset_for_testing_and_validation(args):
    # angles range for dataset
    angles = np.arange(start=0, stop=190, step=30)
    kernal = np.arange(start=9, stop=26, step=2)
    # alpha
    alpha_darkness = np.arange(start=0.05, stop=0.5, step=0.05)
    alpha_brightness = np.arange(start=1.8, stop=2.5, step=0.05)
    # beta
    # beta = np.arange(start=0,stop=1,step=1) # 110 step 10
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='/*.(png|PNG)', printable=False))
    saliency_images_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir, regx='/*.(png|PNG)',
                                                          printable=False))
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir, n_threads=100, mode='RGB')
    SaliencyImages = read_all_imgs(saliency_images_list, path=args.salinecy_data_dir, n_threads=100, mode='RGB2GRAY2')
    maskSaliency = []
    for i in range(len(imagesOrigonal)):
        new_image = np.zeros((final_shape[0], final_shape[1], 3))
        new_mask = np.zeros((final_shape[0], final_shape[1], 1))
        padd_h = int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2)
        padd_w = int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2)
        new_image[padd_h:imagesOrigonal[i].shape[0] + padd_h, padd_w:imagesOrigonal[i].shape[1] + padd_w] = \
            imagesOrigonal[i]
        new_mask[padd_h:imagesOrigonal[i].shape[0] + padd_h, padd_w:imagesOrigonal[i].shape[1] + padd_w] = \
            SaliencyImages[i]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255, gamma)
        (T, saliency) = cv2.threshold(new_mask, 1, 1, cv2.THRESH_BINARY)
        SaliencyImages[i] = saliency
        maskSaliency.append(saliency[:, :, np.newaxis] * imagesOrigonal[i])
        images_list[i] = images_list[i].split('.')[0]

    # now we will go through all of the images and make the dataset
    indexs = np.arange(start=0, stop=len(imagesOrigonal), step=1)
    # TODO need to figure this out (might just want to do the whole image)
    for i in range(len(imagesOrigonal)):
        idx = indexs[i]
        baseImageName = images_list[idx]
        for s in kernal:
            for a in angles:
                # motion blur
                motion_blurred_overlay_img = apply_motion_blur(np.copy(maskSaliency[idx]), s, a)
                blur_mask = apply_motion_blur(np.copy(SaliencyImages[idx]) * 255, s, a)
                motion_blur_mask = np.copy(blur_mask)
                motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                alpha = blur_mask / 255.0
                final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), motion_blurred_overlay_img,
                                                            alpha[:, :, np.newaxis])
                # now save the image
                saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
                saveName = args.output_data_dir + "/gt/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                # cv2.waitKey(0)
                nMask = np.zeros(final_shape)
                nMask[motion_blur_mask == 255] = 64
                cv2.imwrite(saveName, nMask)
                # cv2.imshow('BrightnessImage', nMask)
                # cv2.waitKey(0)
        ### out of focus blur
        for k in kernal:
            focus_blur_overlay_img = create_out_of_focus_blur(np.copy(maskSaliency[idx]), k)
            blur_mask = create_out_of_focus_blur(np.copy(SaliencyImages[idx]) * 255.0, k)
            focus_blur_mask = np.copy(blur_mask)
            focus_blur_mask[np.round(blur_mask, 4) > 0] = 255
            alpha = blur_mask / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), focus_blur_overlay_img,
                                                        alpha[:, :, np.newaxis])
            saveName = args.output_data_dir + "/images/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            # add focus identification to mask
            nMask = np.zeros(final_shape)
            nMask[focus_blur_mask == 255] = 128
            cv2.imwrite(saveName, nMask)
        # next we go to darkness blur
        for a in alpha_darkness:
            # create darkness blur
            dark_blur = create_brightness_and_darkness_blur(np.copy(maskSaliency[idx]), a, 0)
            dark_blur_mask = np.copy(SaliencyImages[idx])
            dark_blur_mask[dark_blur_mask > 0] = 255
            alpha = (np.copy(SaliencyImages[idx]) * 255.0) / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), dark_blur,
                                                        alpha[:, :, np.newaxis])
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            # add darkness blur to mask image
            nMask = np.zeros(final_shape)
            nMask[dark_blur_mask == 255] = 192
            cv2.imwrite(saveName, nMask)
        # next we go to brightness blur
        for a in alpha_brightness:
            # create brightness blur
            bright_blur = create_brightness_and_darkness_blur(np.copy(maskSaliency[idx]), a, 0)
            bright_blur_mask = np.copy(SaliencyImages[idx])
            bright_blur_mask[bright_blur_mask > 0] = 255
            alpha = (np.copy(SaliencyImages[idx]) * 255.0) / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), bright_blur,
                                                        alpha[:, :, np.newaxis])
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image * 1.) / 255, (1 / gamma)) * 255.0)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            # add brightness blur to mask image
            nMask = np.zeros(final_shape)
            nMask[bright_blur_mask == 255] = 255
            cv2.imwrite(saveName, nMask)
    # save control / no blur images
    for n in range(2):
        for i in range(len(imagesOrigonal)):
            # always remember gamma correction
            baseImage = np.power((imagesOrigonal[i] * 1.) / 255, (1 / gamma)) * 255.0
            baseImageName = images_list[i]
            saveName = args.output_data_dir + "/images/" + baseImageName + "_no_blur_" + str(
                n + 1) + args.data_extension
            cv2.imwrite(saveName, baseImage)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_no_blur_" + str(
                n + 1) + args.data_extension
            nMask = np.zeros(final_shape)
            cv2.imwrite(saveName, nMask)

if __name__ == "__main__":
 # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS CREATE BLUR DATASET')
    # directory data location
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_data_dir', type=str)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str,default=".png")
    parser.add_argument('--is_testing', default=False, action='store_true')
    parser.add_argument('--is_chuk_data', default=False, action='store_true')
    parser.add_argument('--salinecy_data_dir',type=str,default='/home/mary/code/local_success_dataset/CHUK_Dataset/Training/salinecy_Images')
    # now resize the data
    # parser.add_argument('--file_input_path', type=str,
    #                 default='/home/mary/code/local_success_dataset/fetch_images/muri_images',
    #                 help='Path to the dataset size 224 by 224 ')
    # parser.add_argument('--file_output_path', type=str,
    #                 default='/home/mary/code/local_success_dataset/BlurDetectionDataset/MURI_DATASET/Training',
    #                 help='Output Path to the dataset for training')
    args = parser.parse_args()
    if args.is_chuk_data:
        if args.is_testing:
            create_chuk_dataset_for_testing_and_validation(args)
        else:
            create_chuk_dataset_for_training(args)
    # elif args.is_testing:
    #     create_syntheic_dataset_for_testing(args)
    # else:
    #     create_syntheic_dataset_for_training(args)
    #   resize_dataset(args)


