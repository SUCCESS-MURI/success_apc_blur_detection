# Author mhatfalv
# Create the blur dataset
import argparse
import copy
import glob
import random

import cv2
import imageio
import numpy as np
from skimage.color import rgb2gray
from skimage.morphology import disk
from scipy.signal import convolve2d

import saliency_detection
from PIL import Image as im, ImageEnhance
import tensorlayer as tl
import matplotlib.pyplot as plt

# create motion blur
# https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
from utils import read_all_imgs

# gamma = 2.5
# gamma = 2.2
# for muri
gamma = 1.5


######## motion blur #########
# https://github.com/Imalne/Defocus-and-Motion-Blur-Detection-with-Deep-Contextual-Features/blob/a368a3e0a8869011ec167bb1f8eb82ceff091e0c/DataCreator/Blend.py#L14
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
    r = random.randint(dmin, dmax)
    return r


def random_motion_blur_kernel(mean=50, variance=15, dmin=10, dmax=100):
    random_degree = normalvariate_random_int(mean, variance, dmin, dmax)
    random_angle = uniform_random_int(-180, 180)
    return random_degree, random_angle

def random_focus_blur_kernel(dmin=10, dmax=50):
    random_kernal_size = uniform_random_int(dmin, dmax)
    return random_kernal_size

##########################################
# we want very extreme values for brightness and darkness since this is causing issues with discriminating
def random_darkness_value(amin=0.01, amax=0.6, bmin=-100, bmax=0):
    alpha = random.uniform(amin, amax)
    beta = random.uniform(bmin, bmax)
    return alpha, beta

def random_brightness_value(amin=1.5, amax=2.6, bmin=0, bmax=100):
    alpha = random.uniform(amin, amax)
    beta = random.uniform(bmin, bmax)
    return alpha, beta

# create out of focus blur for 3 channel images
def apply_out_of_focus_blur(image, kernelsize):
    # create disk kernal
    kernal = disk(kernelsize)
    kernal = kernal / kernal.sum()
    # now convolve
    # https://www.askpython.com/python-modules/opencv-filter2d
    image_blurred = cv2.filter2D(image, -1, kernal)
    return image_blurred

# https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
def create_brightness_and_darkness_blur(image, alpha, beta):
    new_img = image * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img

# https://github.com/Imalne/Defocus-and-Motion-Blur-Detection-with-Deep-Contextual-Features/blob/master/DataCreator/Blend.py#L5
def alpha_blending(defocus, motion, alpha):
    f_defocus = defocus.astype("float32")
    f_motion = motion.astype("float32")
    f_blended = f_defocus * (1 - alpha) + f_motion * alpha
    return f_blended  # .astype("uint8")

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
    return img, y1o, y2o, x1o, x2o

# create training dataset for muri
def create_muri_dataset_for_training(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='/*.(png|PNG)', printable=False))
    images_gt_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir, regx='/*.(png|PNG)', printable=False))
    saliency_images_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='/*.(png|PNG)',printable=False))
    saliency_mask_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir, regx='/*.(png|PNG)',
                                                        printable=False))
    # these images have out of focus blur already
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir + '/', n_threads=100, mode='RGB')
    gtOrigonal = read_all_imgs(images_gt_list, path=args.salinecy_data_dir + '/', n_threads=100, mode='RGB2GRAY2')
    saliencyImages = read_all_imgs(saliency_images_list, path=args.data_dir + '/', n_threads=100,mode='RGB')
    saliencyMask = read_all_imgs(saliency_mask_list, path=args.salinecy_data_dir + '/', n_threads=100,mode='RGB2GRAY2')

    for i in range(len(imagesOrigonal)):
        (T, saliency) = cv2.threshold(gtOrigonal[i], 1, 255, cv2.THRESH_BINARY)
        gtOrigonal[i] = saliency
        if imagesOrigonal[i].shape[2] > 3:
            imagesOrigonal[i] = imagesOrigonal[i][:,:,0:3]
        if imagesOrigonal[i].shape[0] > imagesOrigonal[i].shape[1]:
            imagesOrigonal[i] = cv2.rotate(imagesOrigonal[i], 0)
            gtOrigonal[i] = cv2.rotate(gtOrigonal[i], 0)[:, :, np.newaxis]
        y1, y2 = max(0, int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2)), \
                 min(imagesOrigonal[i].shape[0] + int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2),
                     final_shape[0])
        x1, x2 = max(0, int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2)), \
                 min(imagesOrigonal[i].shape[1] + int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2),
                     final_shape[1])
        y1o, y2o = 0, min(imagesOrigonal[i].shape[0], final_shape[0])
        x1o, x2o = 0, min(imagesOrigonal[i].shape[1], final_shape[1])

        new_image = np.zeros((final_shape[0], final_shape[1], 3))
        new_image[y1:y2, x1:x2] = imagesOrigonal[i][y1o:y2o, x1o:x2o]
        # darkness blur default for 0 pixels
        gtMask = np.ones((final_shape[0], final_shape[1]))
        gtOrigonal[i] = np.squeeze(gtOrigonal[i])
        gtMask[y1:y2, x1:x2] = gtOrigonal[i][y1o:y2o, x1o:x2o]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        images_list[i] = images_list[i].split('.')[0]
        new_mask = np.zeros(gtMask.shape)
        new_mask[gtMask == 0] = 0
        new_mask[gtMask == 1] = 3
        gtOrigonal[i] = new_mask

    for i in range(len(saliencyImages)):
        if saliencyImages[i].shape[2] > 3:
            saliencyImages[i] = saliencyImages[i][:,:,0:3]
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
        saliencyMask[i] = np.squeeze(saliencyMask[i])
        new_mask = np.zeros((final_shape[0], final_shape[1]))
        new_mask[y1:y2, x1:x2] = saliencyMask[i][y1o:y2o, x1o:x2o]
        saliencyImages[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        (T, saliency) = cv2.threshold(new_mask, 1, 1, cv2.THRESH_BINARY)
        saliencyMask[i] = saliency

    # now we will go through all of the images and make the dataset
    idxs = np.arange(start=0, stop=4, step=1)
    saliencyIdx = np.arange(start=0, stop=len(saliencyImages), step=1)
    # this already makes a huge dataset
    for i in range(len(imagesOrigonal)):
        baseImageName = images_list[i]
        np.random.shuffle(idxs)
        for count in range(150):
            final_masked_blurred_image = copy.deepcopy(imagesOrigonal[i])
            nMask = copy.deepcopy(gtOrigonal[i])
            for j in idxs:
                # motion blur
                if j == 0:
                    overlayImageIdx = np.random.choice(saliencyIdx, 1, replace=False)[0]
                    kernal, angle = random_motion_blur_kernel()
                    motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(saliencyImages[overlayImageIdx]),
                                                                   kernal, angle)
                    # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
                    # cv2.waitKey(0)
                    blur_mask = apply_motion_blur(copy.deepcopy(saliencyMask[overlayImageIdx]) * 255, kernal, angle)
                    motion_blur_mask = copy.deepcopy(blur_mask)
                    motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                    alpha = blur_mask / 255.
                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                                         motion_blurred_overlay_img,
                                                                                         placement[1], placement[0],
                                                                                         alpha)
                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    nMask[max(0, placement[0]) + np.argwhere(motion_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 0],
                          max(0, placement[1]) + np.argwhere(motion_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 1
                    # cv2.imshow('BrightnessImage', nMask)
                    # cv2.waitKey(0)
                # Focus blur
                if j == 1:
                    overlayImageIdx = np.random.choice(saliencyIdx, 1, replace=False)[0]
                    kernal = random_focus_blur_kernel()
                    focus_blurred_overlay_img = apply_out_of_focus_blur(copy.deepcopy(saliencyImages[overlayImageIdx]),
                                                                   kernal)
                    # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
                    # cv2.waitKey(0)
                    blur_mask = apply_out_of_focus_blur(copy.deepcopy(saliencyMask[overlayImageIdx]) * 255, kernal)
                    focus_blur_mask = copy.deepcopy(blur_mask)
                    focus_blur_mask[np.round(blur_mask, 4) > 0] = 255
                    alpha = blur_mask / 255.
                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                                         focus_blurred_overlay_img,
                                                                                         placement[1], placement[0],
                                                                                         alpha)
                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    nMask[max(0, placement[0]) + np.argwhere(focus_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 0],
                          max(0, placement[1]) + np.argwhere(focus_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 2
                    # cv2.imshow('BrightnessImage', nMask)
                    # cv2.waitKey(0)
                # darkness blur
                elif j == 2:
                    # create darkness blur
                    overlayImageIdx = np.random.choice(saliencyIdx, 1, replace=False)[0]
                    alpha, beta = random_darkness_value()
                    dark_blurred_overlay_img = create_brightness_and_darkness_blur(
                        copy.deepcopy(saliencyImages[overlayImageIdx]),
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
                                                                                         dark_blurred_overlay_img,
                                                                                         placement[1], placement[0],
                                                                                         alpha_mask)
                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # add darkness blur to mask image
                    # indicator for dark blur
                    nMask[max(0, placement[0]) + np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 0],
                          max(0, placement[1]) + np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 3
                # brightness blur
                elif j == 3:
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
                                                                                         bright_blurred_overlay_img,
                                                                                         placement[1], placement[0],
                                                                                         alpha_mask)
                    # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # add brightness blur to mask image
                    # indicator for brightness
                    nMask[max(0, placement[0]) + np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 0],
                          max(0, placement[1]) + np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 4
            # save final image
            saveName = args.output_data_dir + "/images/" + baseImageName + '_' + str(count) + args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_' + str(count) + args.data_extension
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveNameGt, nMask)

# create testing dataset for muri
def create_muri_dataset_for_testing(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='/*.(png|PNG)', printable=False))
    images_gt_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir, regx='/*.(png|PNG)', printable=False))
    saliency_images_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='/*.(png|PNG)',printable=False))
    saliency_mask_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir, regx='/*.(png|PNG)',
                                                        printable=False))
    # these images have out of focus blur already
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir + '/', n_threads=100, mode='RGB')
    gtOrigonal = read_all_imgs(images_gt_list, path=args.salinecy_data_dir + '/', n_threads=100, mode='RGB2GRAY2')
    saliencyImages = read_all_imgs(saliency_images_list, path=args.data_dir + '/', n_threads=100,mode='RGB')
    saliencyMask = read_all_imgs(saliency_mask_list, path=args.salinecy_data_dir + '/', n_threads=100,mode='RGB2GRAY2')

    for i in range(len(imagesOrigonal)):
        (T, saliency) = cv2.threshold(gtOrigonal[i], 1, 255, cv2.THRESH_BINARY)
        gtOrigonal[i] = saliency
        if imagesOrigonal[i].shape[2] > 3:
            imagesOrigonal[i] = imagesOrigonal[i][:,:,0:3]
        if imagesOrigonal[i].shape[0] > imagesOrigonal[i].shape[1]:
            imagesOrigonal[i] = cv2.rotate(imagesOrigonal[i], 0)
            gtOrigonal[i] = cv2.rotate(gtOrigonal[i], 0)[:, :, np.newaxis]
        y1, y2 = max(0, int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2)), \
                 min(imagesOrigonal[i].shape[0] + int((final_shape[0] + 1 - imagesOrigonal[i].shape[0]) / 2),
                     final_shape[0])
        x1, x2 = max(0, int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2)), \
                 min(imagesOrigonal[i].shape[1] + int((final_shape[1] + 1 - imagesOrigonal[i].shape[1]) / 2),
                     final_shape[1])
        y1o, y2o = 0, min(imagesOrigonal[i].shape[0], final_shape[0])
        x1o, x2o = 0, min(imagesOrigonal[i].shape[1], final_shape[1])

        new_image = np.zeros((final_shape[0], final_shape[1], 3))
        new_image[y1:y2, x1:x2] = imagesOrigonal[i][y1o:y2o, x1o:x2o]
        # darkness blur default for 0 pixels
        gtMask = np.ones((final_shape[0], final_shape[1]))
        gtOrigonal[i] = np.squeeze(gtOrigonal[i])
        gtMask[y1:y2, x1:x2] = gtOrigonal[i][y1o:y2o, x1o:x2o]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        images_list[i] = images_list[i].split('.')[0]
        new_mask = np.zeros(gtMask.shape)
        new_mask[gtMask == 0] = 0
        new_mask[gtMask == 1] = 3
        gtOrigonal[i] = new_mask

    for i in range(len(saliencyImages)):
        if saliencyImages[i].shape[2] > 3:
            saliencyImages[i] = saliencyImages[i][:,:,0:3]
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
        saliencyMask[i] = np.squeeze(saliencyMask[i])
        new_mask = np.zeros((final_shape[0], final_shape[1]))
        new_mask[y1:y2, x1:x2] = saliencyMask[i][y1o:y2o, x1o:x2o]
        saliencyImages[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        (T, saliency) = cv2.threshold(new_mask, 1, 1, cv2.THRESH_BINARY)
        saliencyMask[i] = saliency

    # now we will go through all of the images and make the dataset
    # save baseline images
    for i in range(len(imagesOrigonal)):
        # always remember gamma correction
        baseImage = np.round(np.power((imagesOrigonal[i] * 1.) / 255, (1 / gamma)) * 255.0).astype(np.uint8)
        baseImageName = images_list[i]
        saveName = args.output_data_dir + "/images/" + baseImageName + '_noblur' + args.data_extension
        imageio.imsave(saveName, baseImage)
        # get ground truth mask
        saveName = args.output_data_dir + "/gt/" + baseImageName + '_noblur' + args.data_extension
        nMask = np.zeros(final_shape)
        gt = np.squeeze(gtOrigonal[i])
        nMask[gt == 1] = 64
        nMask[gt == 2] = 128
        nMask[gt == 3] = 192
        nMask[gt == 4] = 255
        cv2.imwrite(saveName, nMask)

    # this already makes a huge dataset
    for i in range(len(imagesOrigonal)):
        baseImageName = images_list[i]
        for count in range(10):
            nMask = copy.deepcopy(gtOrigonal[i])
            # motion blur
            kernal, angle = random_motion_blur_kernel()
            motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(saliencyImages[i]),kernal, angle)
            blur_mask = apply_motion_blur(copy.deepcopy(saliencyMask[i]) * 255, kernal, angle)
            motion_blur_mask = copy.deepcopy(blur_mask)
            motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
            alpha = blur_mask / 255.
            final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(copy.deepcopy(imagesOrigonal[i]),
                                                                                 motion_blurred_overlay_img,
                                                                                 0, 0, alpha)
            # cv2.imshow('MotionImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            nMask[np.argwhere(motion_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 0],
                  np.argwhere(motion_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 1
            saveName = args.output_data_dir + "/images/motion_" + baseImageName + '_' + str(count) + args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                                           / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/motion_" + baseImageName + '_' + str(count) + args.data_extension
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveNameGt, nMask)

            # Focus blur
            nMask = copy.deepcopy(gtOrigonal[i])
            kernal = random_focus_blur_kernel()
            focus_blurred_overlay_img = apply_out_of_focus_blur(copy.deepcopy(saliencyImages[i]),kernal)
            blur_mask = apply_out_of_focus_blur(copy.deepcopy(saliencyMask[i]) * 255, kernal)
            focus_blur_mask = copy.deepcopy(blur_mask)
            focus_blur_mask[np.round(blur_mask, 4) > 0] = 255
            alpha = blur_mask / 255.
            final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(copy.deepcopy(imagesOrigonal[i]),
                                                                                 focus_blurred_overlay_img,
                                                                                 0, 0, alpha)
            # cv2.imshow('MotionImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            nMask[np.argwhere(focus_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 0],
                  np.argwhere(focus_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 2
            saveName = args.output_data_dir + "/images/focus_" + baseImageName + '_' + str(count) + args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                                           / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/focus_" + baseImageName + '_' + str(count) + args.data_extension
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveNameGt, nMask)
            # darkness blur
            # create darkness blur
            nMask = copy.deepcopy(gtOrigonal[i])
            alpha, beta = random_darkness_value()
            dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[i]),alpha, beta)
            # cv2.imshow('BrightnessImage', dark_blur)
            # cv2.waitKey(0)
            dark_blur_mask = copy.deepcopy(saliencyMask[i])
            dark_blur_mask[dark_blur_mask > 0] = 255
            alpha_mask = (copy.deepcopy(saliencyMask[i]) * 255.0) / 255.
            final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(copy.deepcopy(imagesOrigonal[i]),
                                                                                 dark_blurred_overlay_img,
                                                                                 0, 0,alpha_mask)
            # cv2.imshow('DarknessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # add darkness blur to mask image
            # indicator for dark blur
            nMask[np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 0],
                  np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 3
            saveName = args.output_data_dir + "/images/darkness_" + baseImageName + '_' + str(count) + \
                       args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                                           / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/darkness_" + baseImageName + '_' + str(count) + args.data_extension
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveNameGt, nMask)
            # brightness blur
            alpha, beta = random_brightness_value()
            bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[i]), alpha,
                                                                             beta)
            # cv2.imshow('BrightnessImage', dark_blur)
            # cv2.waitKey(0)
            nMask = copy.deepcopy(gtOrigonal[i])
            bright_blur_mask = copy.deepcopy(saliencyMask[i])
            bright_blur_mask[bright_blur_mask > 0] = 255
            alpha_mask = (copy.deepcopy(saliencyMask[i]) * 255.0) / 255.
            final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(copy.deepcopy(imagesOrigonal[i]),
                                                                                 bright_blurred_overlay_img,
                                                                                 0, 0,alpha_mask)
            # cv2.imshow('BrightnessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # add brightness blur to mask image
            # indicator for brightness
            nMask[np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 0],
                  np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 4
            # save final image
            saveName = args.output_data_dir + "/images/brightness_" + baseImageName + '_' + str(count) + \
                       args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/brightness_" + baseImageName + '_' + str(count) + \
                         args.data_extension
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveNameGt, nMask)

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS MURI CREATE BLUR DATASET')
    # directory data location
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_data_dir', type=str)
    parser.add_argument('--salinecy_data_dir', type=str, default=None)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str, default=".png")
    parser.add_argument('--is_testing', default=False, action='store_true')
    parser.add_argument('--is_muri_dataset', default=False, action='store_true')
    parser.add_argument('--is_sensitivity', default=False, action='store_true')
    args = parser.parse_args()
    if args.is_muri_dataset and not args.is_testing:
        create_muri_dataset_for_training(args)
    elif args.is_muri_dataset and args.is_testing:
        create_muri_dataset_for_testing(args)
