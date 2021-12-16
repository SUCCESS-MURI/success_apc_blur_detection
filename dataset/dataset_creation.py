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
from scipy.ndimage import convolve

import saliency_detection
from PIL import Image as im, ImageEnhance
import tensorlayer as tl
import matplotlib.pyplot as plt

# create motion blur
# https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
from utils import read_all_imgs

#gamma = 2.5
gamma = 2.2
# for muri
# gamma = 1.766666667

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
# we want very extreme values for brightness and darkness since this is causing issues with discriminating
def random_darkness_value(amin=0.01,amax=0.6,bmin=-200,bmax=-40):
    alpha = random.uniform(amin,amax)
    beta = random.uniform(bmin,bmax)
    return alpha,beta

def random_brightness_value(amin=1.9,amax=2.6,bmin=40,bmax=200):
    alpha = random.uniform(amin,amax)
    beta = random.uniform(bmin,bmax)
    return alpha,beta

def random_focus_blur_kernel(dmin=3, dmax=50):
    random_kernal_size = uniform_random_int(dmin, dmax)
    return random_kernal_size

# create out of focus blur for 3 channel images
def create_out_of_focus_blur(image,kernelsize):
    kernal = disk(kernelsize)
    kernal = kernal/kernal.sum()
    image_blurred = np.stack([convolve(c,kernal) for c in image.T]).T
    #image_blurred = pyblur.DefocusBlur(image,kernelsize)#cv2.blur(image,(kernelsize,kernelsize))
    return image_blurred

# https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
def create_brightness_and_darkness_blur(image, alpha, beta):
    new_img = image * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img

# we define brightness and darkness blur by thesholding the image
def define_brightness_and_darkness_blur(image,mask):
    # scale to 0.0 to 1.0
    grayscale = rgb2gray(image)/255.0
    # we only want to label extreme values
    mask[grayscale > 0.99] = 4
    mask[grayscale < 0.001] = 3

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

def create_cuhk_dataset_for_training(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir + '/images', regx='/*.(jpg|JPG)', printable=False))
    images_gt_list = sorted(tl.files.load_file_list(path=args.data_dir + '/gt', regx='/*.(png|PNG)', printable=False))
    saliency_images_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir + '/images', regx='/*.(jpg|JPG)',
                                                          printable=False))
    saliency_mask_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir + '/gt', regx='/*.(png|PNG)',
                                                        printable=False))
    # these images have out of focus blur already
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir + '/images/', n_threads=100, mode='RGB')
    gtOrigonal = read_all_imgs(images_gt_list, path=args.data_dir + '/gt/', n_threads=100, mode='RGB2GRAY2')
    saliencyImages = read_all_imgs(saliency_images_list, path=args.salinecy_data_dir + '/images/', n_threads=100,
                                   mode='RGB')
    saliencyMask = read_all_imgs(saliency_mask_list, path=args.salinecy_data_dir + '/gt/', n_threads=100,
                                 mode='RGB2GRAY2')

    for i in range(len(imagesOrigonal)):
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
        # darkness blur deafult for 0 pixels
        gtMask = np.ones((final_shape[0], final_shape[1], 1))
        gtMask[y1:y2, x1:x2] = gtOrigonal[i][y1o:y2o, x1o:x2o]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        images_list[i] = images_list[i].split('.')[0]
        new_mask = np.zeros(gtMask.shape)
        new_mask[gtMask == 0] = 0
        new_mask[gtMask == 1] = 3
        new_mask[gtMask == 255] = 2
        # define brightness and darkness in images
        define_brightness_and_darkness_blur(imagesOrigonal[i],new_mask)
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
                    kernal, angle = random_motion_blur_kernel()
                    motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(saliencyImages[overlayImageIdx]),
                                                                   kernal, angle)
                    # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
                    # cv2.waitKey(0)
                    maskdummy = np.squeeze(np.zeros(nMask.shape))
                    define_brightness_and_darkness_blur(motion_blurred_overlay_img,maskdummy)
                    #blur_mask = copy.deepcopy(saliencyMask[overlayImageIdx])*255
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
                    maskdummy[motion_blur_mask != 255] = 255
                    nMask[max(0, placement[0]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 3)[:, 0],
                          max(0, placement[1]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 3)[:, 1]] = 3
                    nMask[max(0, placement[0]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 4)[:, 0],
                          max(0, placement[1]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 4)[:, 1]] = 4
                    # cv2.imshow('BrightnessImage', nMask)
                    # cv2.waitKey(0)
                # darkness blur
                elif j == 1:
                    # create darkness blur
                    overlayImageIdx = np.random.choice(saliencyIdx, 1, replace=False)[0]
                    alpha, beta = random_darkness_value()
                    dark_blurred_overlay_img = create_brightness_and_darkness_blur(
                        copy.deepcopy(saliencyImages[overlayImageIdx]),
                        alpha, beta)
                    # cv2.imshow('BrightnessImage', dark_blur)
                    # cv2.waitKey(0)
                    maskdummy = np.squeeze(np.zeros(nMask.shape))
                    define_brightness_and_darkness_blur(dark_blurred_overlay_img, maskdummy)

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
                          max(0, placement[1]) + np.argwhere(dark_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 0 # 3
                    maskdummy[dark_blur_mask != 255] = 255
                    nMask[max(0, placement[0]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 3)[:, 0],
                          max(0, placement[1]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 3)[:, 1]] = 3
                    nMask[max(0, placement[0]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 4)[:, 0],
                          max(0, placement[1]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 4)[:, 1]] = 4
                # brightness blur
                elif j == 2:
                    # create brightness blur
                    overlayImageIdx = np.random.choice(saliencyIdx, 1, replace=False)[0]
                    alpha, beta = random_brightness_value()
                    bright_blurred_overlay_img = create_brightness_and_darkness_blur(
                        copy.deepcopy(saliencyImages[overlayImageIdx]), alpha, beta)
                    # cv2.imshow('BrightnessImage', dark_blur)
                    # cv2.waitKey(0)
                    maskdummy = np.squeeze(np.zeros(nMask.shape))
                    define_brightness_and_darkness_blur(bright_blurred_overlay_img, maskdummy)
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
                          max(0, placement[1]) + np.argwhere(bright_blur_mask[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 0 # 4
                    maskdummy[bright_blur_mask != 255] = 255
                    nMask[max(0, placement[0]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 3)[:, 0],
                          max(0, placement[1]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 3)[:, 1]] = 3
                    nMask[max(0, placement[0]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 4)[:, 0],
                          max(0, placement[1]) + np.argwhere(maskdummy[y1o:y2o, x1o:x2o] == 4)[:, 1]] = 4
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

def create_cuhk_dataset_for_testing(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir + '/image', regx='/*.(jpg|JPG)', printable=False))
    images_gt_list = sorted(tl.files.load_file_list(path=args.data_dir + '/gt', regx='/*.(png|PNG)', printable=False))
    # these images have out of focus blur already
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir + '/image/', n_threads=100, mode='RGB')
    gtOrigonal = read_all_imgs(images_gt_list, path=args.data_dir + '/gt/', n_threads=100, mode='RGB2GRAY2')
    # saliencyImages = copy.deepcopy(imagesOrigonal)
    # saliencyMask = copy.deepcopy(gtOrigonal)

    for i in range(len(imagesOrigonal)):
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
        # darkness blur deafult for 0 pixels
        gtMask = np.ones((final_shape[0], final_shape[1], 1))
        gtMask[y1:y2, x1:x2] = gtOrigonal[i][y1o:y2o, x1o:x2o]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        images_list[i] = images_list[i].split('.')[0]
        new_mask = np.zeros(gtMask.shape)
        new_mask[gtMask == 0] = 0
        new_mask[gtMask == 1] = 3
        if 'motion' in images_list[i]:
            new_mask[gtMask == 255] = 1
        else:
            new_mask[gtMask == 255] = 2
        # define brightness and darkness in images
        define_brightness_and_darkness_blur(imagesOrigonal[i], new_mask)
        gtOrigonal[i] = new_mask

    # save baseline images
    count = 0
    for i in range(len(imagesOrigonal)):
        # always remember gamma correction
        baseImage = np.round(np.power((imagesOrigonal[i] * 1.) / 255, (1 / gamma)) * 255.0).astype(np.uint8)
        baseImageName = images_list[i]
        saveName = args.output_data_dir + "/images/" + baseImageName + '_' + str(count) + args.data_extension
        imageio.imsave(saveName, baseImage)
        # get ground truth mask
        saveName = args.output_data_dir + "/gt/" + baseImageName + '_' + str(count) + args.data_extension
        nMask = np.zeros(final_shape)
        gt = np.squeeze(gtOrigonal[i])
        nMask[gt == 1] = 64
        nMask[gt == 2] = 128
        nMask[gt == 3] = 192
        nMask[gt == 4] = 255
        cv2.imwrite(saveName, nMask)

    # now we will go through all of the images and make the dataset to make brightness and darkness blurs
    # this already makes a huge dataset
    count = 1
    for i in range(len(imagesOrigonal)):
        baseImageName = images_list[i]
        # https://www.programiz.com/python-programming/examples/odd-even
        if i % 2 == 0:
            # create darkness blur
            alpha, beta = random_darkness_value()
            dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(imagesOrigonal[i]), alpha,
                                                                           beta)
            # cv2.imshow('BrightnessImage', dark_blur)
            # cv2.waitKey(0)
            # indicator for dark blur
            #nMask = np.ones(final_shape) * 3
            nMask = np.zeros(final_shape)
            define_brightness_and_darkness_blur(dark_blurred_overlay_img, nMask)
            # cv2.imshow('DarknessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # save just the darkness
            # save final image
            saveName = args.output_data_dir + "/images/darkness_" + baseImageName + '_' + str(count) + args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(dark_blurred_overlay_img)[:, :, 0:3] * 1.)
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
        else:
            # create brightness blur
            alpha, beta = random_brightness_value()
            image = copy.deepcopy(imagesOrigonal[i])
            bright_blurred_overlay_img = create_brightness_and_darkness_blur(image, alpha, beta)
            # cv2.imshow('BrightnessImage', dark_blur)
            # cv2.waitKey(0)
            # indicator for brightness
            #nMask = np.ones(final_shape) * 4
            nMask = np.zeros(final_shape)
            define_brightness_and_darkness_blur(bright_blurred_overlay_img, nMask)
            # save final image
            saveName = args.output_data_dir + "/images/brightness_" + baseImageName + '_' + str(count) + args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(bright_blurred_overlay_img)[:, :, 0:3] * 1.)
                                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/brightness_" + baseImageName + '_' + str(count) + args.data_extension
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveNameGt, nMask)

# def create_cuhk_dataset_for_sensitivity(args):
#     motion_degree = np.array([30,60,90,120,150,180])
#     motion_kernal_size = np.array([3,5,7,9,25,33,47])
#     focus_kernal_size = np.array([3,5,7,9,25,33,47])
#     darkness_alpha = np.array([0.1,0.4,0.6,0.9])
#     darkness_beta = np.array([0,-50,-100,-200])
#     brightness_alpha = np.array([1.1,1.4,1.6,1.9])
#     brightness_beta = np.array([0,50,100,200])
#     # all ranges of motion, focus, darkness and brightness images are created
#     final_shape = (480, 640)
#     # make it with respect to aspect ratio
#     scale = 30
#     width = int(480 * scale / 100)
#     height = int(640 * scale / 100)
#     half_shape = (width, height)
#     scale = 60
#     width = int(480 * scale / 100)
#     height = int(640 * scale / 100)
#     small_shape = (width, height)
#     tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
#     tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
#     # list of all original Images
#     images_list = sorted(tl.files.load_file_list(path=args.data_dir + '/image', regx='/*.(jpg|JPG)', printable=False))
#     images_gt_list = sorted(tl.files.load_file_list(path=args.data_dir + '/gt', regx='/*.(png|PNG)', printable=False))
#     # these images have out of focus blur already
#     imagesOrigonal = read_all_imgs(images_list, path=args.data_dir + '/image/', n_threads=100, mode='RGB')
#     gtOrigonal = read_all_imgs(images_gt_list, path=args.data_dir + '/gt/', n_threads=100, mode='RGB2GRAY2')
#     saliencyImages = copy.deepcopy(imagesOrigonal)
#     saliencyMask = copy.deepcopy(gtOrigonal)
#     origionalSaliencyMask = copy.deepcopy(saliencyMask)
#
#     for i in range(len(saliencyImages)):
#         if saliencyImages[i].shape[0] > saliencyImages[i].shape[1]:
#             saliencyImages[i] = cv2.rotate(saliencyImages[i], 0)
#             saliencyMask[i] = cv2.rotate(saliencyMask[i], 0)[:, :, np.newaxis]
#         y1, y2 = max(0, int((final_shape[0] + 1 - saliencyImages[i].shape[0]) / 2)), \
#                  min(saliencyImages[i].shape[0] + int((final_shape[0] + 1 - saliencyImages[i].shape[0]) / 2),
#                          final_shape[0])
#         x1, x2 = max(0, int((final_shape[1] + 1 - saliencyImages[i].shape[1]) / 2)), \
#                 min(saliencyImages[i].shape[1] + int((final_shape[1] + 1 - saliencyImages[i].shape[1]) / 2),
#                          final_shape[1])
#         y1o, y2o = 0, min(saliencyImages[i].shape[0], final_shape[0])
#         x1o, x2o = 0, min(saliencyImages[i].shape[1], final_shape[1])
#
#         new_image = np.zeros((final_shape[0], final_shape[1], 3))
#         new_image[y1:y2, x1:x2] = saliencyImages[i][y1o:y2o, x1o:x2o]
#         gtMask = np.ones((final_shape[0], final_shape[1], 1))
#         gtMask[y1:y2, x1:x2] = saliencyMask[i][y1o:y2o, x1o:x2o]
#         saliencyImages[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
#         new_mask = np.zeros(gtMask.shape)
#         new_mask[gtMask == 0] = 0
#         new_mask[gtMask == 1] = 3
#         images_list[i] = images_list[i].split('.')[0]
#         if 'motion' in images_list[i]:
#             new_mask[gtMask == 255] = 1
#         else:
#             new_mask[gtMask == 255] = 2
#         mask = np.zeros(new_mask.shape)
#         mask[new_mask > 0] = 1
#         saliencyMask[i] = np.logical_not(mask) * 1.0
#         # # define brightness and darkness in images
#         define_brightness_and_darkness_blur(saliencyImages[i], new_mask)
#         origionalSaliencyMask[i] = new_mask
#     # now we will go through all of the images and make the dataset to make brightness and darkness blurs
#     index = np.arange(0,len(saliencyImages),1)
#     np.random.shuffle(index)
#     # this already makes a huge dataset
#     # for i in range(100):
#     #     baseImageName = images_list[i]
#     #     if 'focus' in baseImageName:
#     #         for angle in motion_degree:
#     #             for kernal_size in motion_kernal_size:
#     #                 # create motion blur
#     #                 motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(saliencyImages[i]), kernal_size,
#     #                                                                    angle)
#     #                 # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
#     #                 # cv2.waitKey(0)
#     #                 # Define brightness and darkness in image
#     #                 maskdummy = np.squeeze(np.zeros(final_shape))
#     #                 define_brightness_and_darkness_blur(motion_blurred_overlay_img, maskdummy)
#     #
#     #                 blur_mask = apply_motion_blur(copy.deepcopy(saliencyMask[i]) * 255, kernal_size, angle)
#     #                 motion_blur_mask = copy.deepcopy(blur_mask)
#     #                 motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
#     #                 alpha_mask = motion_blur_mask / 255.
#     #                 final_masked_blurred_image = alpha_blending(copy.deepcopy(saliencyImages[i]),
#     #                                                                 motion_blurred_overlay_img,
#     #                                                                 alpha_mask[:, :, np.newaxis])
#     #                 # cv2.imshow('MotionImage', final_masked_blurred_image)
#     #                 # cv2.waitKey(0)
#     #                 nMask = copy.deepcopy(origionalSaliencyMask[i])
#     #                 #define_brightness_and_darkness_blur(saliencyImages[i], nMask)
#     #                 nMask[motion_blur_mask == 255] = 1
#     #                 maskdummy[np.squeeze(motion_blur_mask) != 255] = 255
#     #                 nMask[maskdummy == 3] = 3
#     #                 nMask[maskdummy == 4] = 4
#     #                 nMask[nMask == 1] = 64
#     #                 nMask[nMask == 2] = 128
#     #                 nMask[nMask == 3] = 192
#     #                 nMask[nMask == 4] = 255
#     #                 saveName = args.output_data_dir + "/images/" + baseImageName + '_motion_angle_' + str(angle) + \
#     #                                "_kernal_size_" + str(kernal_size) + args.data_extension
#     #                 final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#     #                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#     #                 # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#     #                 imageio.imwrite(saveName, final_masked_blurred_image)
#     #                 # create and save ground truth mask
#     #                 saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_motion_angle_' + str(angle) + \
#     #                                  "_kernal_size_" + str(kernal_size) + args.data_extension
#     #                 cv2.imwrite(saveNameGt, nMask)
#     #     else:
#     #         # create out of focus blur
#     #         for kernal_size in focus_kernal_size:
#     #             focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(saliencyImages[i]), kernal_size)
#     #
#     #             # # Define brightness and darkness in image
#     #             maskdummy = np.squeeze(np.zeros(final_shape))
#     #             define_brightness_and_darkness_blur(focus_blurred_overlay_img, maskdummy)
#     #
#     #             # now make the mask
#     #             blur_mask = create_out_of_focus_blur(copy.deepcopy(saliencyMask[i]) * 255., kernal_size)
#     #             focus_blur_mask = copy.deepcopy(blur_mask)
#     #             focus_blur_mask[np.round(blur_mask / 255., 1) > 0] = 255
#     #             alpha_mask = focus_blur_mask / 255.
#     #             final_masked_blurred_image = alpha_blending(copy.deepcopy(saliencyImages[i]),focus_blurred_overlay_img,
#     #                                                             alpha_mask)
#     #             nMask = copy.deepcopy(origionalSaliencyMask[i])
#     #             #define_brightness_and_darkness_blur(saliencyImages[i], nMask)
#     #             nMask[focus_blur_mask == 255] = 2
#     #             maskdummy[np.squeeze(focus_blur_mask) != 255] = 255
#     #             nMask[maskdummy == 3] = 3
#     #             nMask[maskdummy == 4] = 4
#     #             nMask[nMask == 1] = 64
#     #             nMask[nMask == 2] = 128
#     #             nMask[nMask == 3] = 192
#     #             nMask[nMask == 4] = 255
#     #             saveName = args.output_data_dir + "/images/" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
#     #                            + args.data_extension
#     #             final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#     #                              / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#     #             # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#     #             imageio.imwrite(saveName, final_masked_blurred_image)
#     #             # create and save ground truth mask
#     #             saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
#     #                              + args.data_extension
#     #             cv2.imwrite(saveNameGt, nMask)
#     #     if i % 2 == 0:
#     #         # create darkness blur
#     #         for alpha in darkness_alpha:
#     #             for beta in darkness_beta:
#     #                 dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[i]),
#     #                                                                                    alpha, beta)
#     #                 # # Define brightness and darkness in image
#     #                 # maskdummy = np.squeeze(np.zeros(final_shape))
#     #                 # define_brightness_and_darkness_blur(dark_blurred_overlay_img, maskdummy)
#     #                 # cv2.imshow('BrightnessImage', dark_blur)
#     #                 # cv2.waitKey(0)
#     #                 # indicator for dark blur
#     #                 dark_mask = copy.deepcopy(saliencyMask[i]) * 255
#     #                 dark_mask[np.round(dark_mask, 4) > 0] = 255
#     #                 alpha_mask = dark_mask / 255.
#     #                 nMask = np.zeros(final_shape)
#     #                 define_brightness_and_darkness_blur(dark_blurred_overlay_img, nMask)
#     #                 #nMask[dark_mask == 255] = 3
#     #                 # maskdummy[np.squeeze(dark_mask) != 255] = 255
#     #                 # nMask[maskdummy == 3] = 3
#     #                 # nMask[maskdummy == 4] = 4
#     #                 nMask[nMask == 1] = 64
#     #                 nMask[nMask == 2] = 128
#     #                 nMask[nMask == 3] = 192
#     #                 nMask[nMask == 4] = 255
#     #                 final_masked_blurred_image = alpha_blending(copy.deepcopy(saliencyImages[i]),dark_blurred_overlay_img,
#     #                                                                 alpha_mask)
#     #                 # cv2.imshow('DarknessImage', final_masked_blurred_image)
#     #                 # cv2.waitKey(0)
#     #                 # save just the darkness
#     #                 # save final image
#     #                 saveName = args.output_data_dir + "/images/" + baseImageName + '_darkness_alpha_' + str(alpha) + \
#     #                                "_beta_" + str(beta) + args.data_extension
#     #                 final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#     #                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#     #                 # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#     #                 imageio.imwrite(saveName, final_masked_blurred_image)
#     #                 # create and save ground truth mask
#     #                 saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_darkness_alpha_' + str(alpha) + \
#     #                                  "_beta_" + str(beta) + args.data_extension
#     #                 cv2.imwrite(saveNameGt, nMask)
#     #     else:
#     #         # create brightness blur
#     #         for alpha in brightness_alpha:
#     #             for beta in brightness_beta:
#     #                 bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[i]),
#     #                         alpha, beta)
#     #                 # # Define brightness and darkness in image
#     #                 # maskdummy = np.squeeze(np.zeros(final_shape))
#     #                 # define_brightness_and_darkness_blur(bright_blurred_overlay_img, maskdummy)
#     #                 # cv2.imshow('BrightnessImage', dark_blur)
#     #                 # cv2.waitKey(0)
#     #                 # indicator for bright blur
#     #                 bright_mask = copy.deepcopy(saliencyMask[i]) * 255
#     #                 bright_mask[np.round(bright_mask, 4) > 0] = 255
#     #                 alpha_mask = bright_mask / 255.
#     #                 nMask = np.zeros(final_shape)
#     #                 define_brightness_and_darkness_blur(bright_blurred_overlay_img, nMask)
#     #                 nMask[nMask == 1] = 64
#     #                 nMask[nMask == 2] = 128
#     #                 nMask[nMask == 3] = 192
#     #                 nMask[nMask == 4] = 255
#     #                 final_masked_blurred_image = alpha_blending(copy.deepcopy(saliencyImages[i]),bright_blurred_overlay_img,
#     #                                                                 alpha_mask)
#     #                 # cv2.imshow('DarknessImage', final_masked_blurred_image)
#     #                 # cv2.waitKey(0)
#     #                 # save final image
#     #                 saveName = args.output_data_dir + "/images/" + baseImageName + '_brightness_alpha_' + str(alpha) + \
#     #                                "_beta_" + str(beta) + args.data_extension
#     #                 final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#     #                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#     #                 # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#     #                 imageio.imwrite(saveName, final_masked_blurred_image)
#     #                 # create and save ground truth mask
#     #                 saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_brightness_alpha_' + str(alpha) + \
#     #                                  "_beta_" + str(beta) + args.data_extension
#     #                 cv2.imwrite(saveNameGt, nMask)
#
#     count = 1
#     # now resize the images to half of the current resolution
#     for ind in range(40):
#         i = index[ind]
#         baseImageName = images_list[i]
#         # trying to get 20 focus and 20 motion images even count
#         if 'focus' in baseImageName and count > 40:
#             continue
#         baseImageResized = cv2.resize(saliencyImages[i],(half_shape[1],half_shape[0]),interpolation=cv2.INTER_NEAREST)
#         saliencyMaskResized = np.round(cv2.resize(saliencyMask[i],(half_shape[1],half_shape[0]),
#                                                   interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis])
#         origionalMaskResized = np.round(cv2.resize(origionalSaliencyMask[i],(half_shape[1],half_shape[0]),
#                                                    interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis])
#         if 'focus' in baseImageName:
#             for angle in motion_degree:
#                 for kernal_size in motion_kernal_size:
#                     # create motion blur
#                     motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(baseImageResized), kernal_size,
#                                                                        angle)
#                     # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
#                     # cv2.waitKey(0)
#                     # Define brightness and darkness in image
#                     maskdummy = np.squeeze(np.zeros(half_shape))
#                     define_brightness_and_darkness_blur(motion_blurred_overlay_img, maskdummy)
#
#                     blur_mask = apply_motion_blur(copy.deepcopy(saliencyMaskResized) * 255, kernal_size, angle)
#                     motion_blur_mask = copy.deepcopy(blur_mask)
#                     motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
#                     alpha_mask = motion_blur_mask / 255.
#                     final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
#                                                                     motion_blurred_overlay_img,
#                                                                     alpha_mask[:, :, np.newaxis])
#                     # cv2.imshow('MotionImage', final_masked_blurred_image)
#                     # cv2.waitKey(0)
#                     nMask = copy.deepcopy(origionalMaskResized)
#                     #define_brightness_and_darkness_blur(saliencyImages[i], nMask)
#                     nMask[motion_blur_mask == 255] = 1
#                     maskdummy[np.squeeze(motion_blur_mask) != 255] = 255
#                     nMask[maskdummy == 3] = 3
#                     nMask[maskdummy == 4] = 4
#                     nMask[nMask == 1] = 64
#                     nMask[nMask == 2] = 128
#                     nMask[nMask == 3] = 192
#                     nMask[nMask == 4] = 255
#                     saveName = args.output_data_dir + "/images/Resized_halforigional_" + baseImageName + \
#                                '_motion_angle_' + str(angle) + "_kernal_size_" + str(kernal_size) + args.data_extension
#                     final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#                                      / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#                     # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#                     imageio.imwrite(saveName, final_masked_blurred_image)
#                     # create and save ground truth mask
#                     saveNameGt = args.output_data_dir + "/gt/Resized_halforigional_" + baseImageName + \
#                                  '_motion_angle_' + str(angle) + "_kernal_size_" + str(kernal_size) + \
#                                  args.data_extension
#                     imageio.imwrite(saveNameGt, nMask)
#         else:
#             # create out of focus blur
#             for kernal_size in focus_kernal_size:
#                 focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(baseImageResized), kernal_size)
#
#                 # # Define brightness and darkness in image
#                 maskdummy = np.squeeze(np.zeros(half_shape))
#                 define_brightness_and_darkness_blur(focus_blurred_overlay_img, maskdummy)
#
#                 # now make the mask
#                 blur_mask = create_out_of_focus_blur(copy.deepcopy(saliencyMaskResized) * 255., kernal_size)
#                 focus_blur_mask = copy.deepcopy(blur_mask)
#                 focus_blur_mask[np.round(blur_mask / 255., 1) > 0] = 255
#                 alpha_mask = focus_blur_mask / 255.
#                 final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),focus_blurred_overlay_img,
#                                                                 alpha_mask)
#                 nMask = copy.deepcopy(origionalMaskResized)
#                 #define_brightness_and_darkness_blur(saliencyImages[i], nMask)
#                 nMask[focus_blur_mask == 255] = 2
#                 maskdummy[np.squeeze(focus_blur_mask) != 255] = 255
#                 nMask[maskdummy == 3] = 3
#                 nMask[maskdummy == 4] = 4
#                 nMask[nMask == 1] = 64
#                 nMask[nMask == 2] = 128
#                 nMask[nMask == 3] = 192
#                 nMask[nMask == 4] = 255
#                 saveName = args.output_data_dir + "/images/Resized_halforigional_" + baseImageName + \
#                            '_focus_kernal_size_' + str(kernal_size) + args.data_extension
#                 final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#                 # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#                 imageio.imwrite(saveName, final_masked_blurred_image)
#                 # create and save ground truth mask
#                 saveNameGt = args.output_data_dir + "/gt/Resized_halforigional_" + baseImageName + \
#                              '_focus_kernal_size_' + str(kernal_size) + args.data_extension
#                 imageio.imwrite(saveNameGt, nMask)
#                 count += 1
#         if i % 2 == 0:
#             # create darkness blur
#             for alpha in darkness_alpha:
#                 for beta in darkness_beta:
#                     dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
#                                                                                        alpha, beta)
#                     # Define brightness and darkness in image
#                     # maskdummy = np.squeeze(np.zeros(half_shape))
#                     # define_brightness_and_darkness_blur(dark_blurred_overlay_img, maskdummy)
#                     # cv2.imshow('BrightnessImage', dark_blur)
#                     # cv2.waitKey(0)
#                     # indicator for dark blur
#                     dark_mask = copy.deepcopy(saliencyMaskResized) * 255
#                     dark_mask[np.round(dark_mask, 4) > 0] = 255
#                     alpha_mask = dark_mask / 255.
#                     nMask = np.zeros(half_shape)
#                     define_brightness_and_darkness_blur(dark_blurred_overlay_img, nMask)
#                     nMask[nMask == 1] = 64
#                     nMask[nMask == 2] = 128
#                     nMask[nMask == 3] = 192
#                     nMask[nMask == 4] = 255
#                     final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),dark_blurred_overlay_img,
#                                                                     alpha_mask)
#                     # cv2.imshow('DarknessImage', final_masked_blurred_image)
#                     # cv2.waitKey(0)
#                     # save just the darkness
#                     # save final image
#                     saveName = args.output_data_dir + "/images/Resized_halforigional_" + baseImageName + \
#                                '_darkness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
#                     final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#                                      / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#                     # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#                     imageio.imwrite(saveName, final_masked_blurred_image)
#                     # create and save ground truth mask
#                     saveNameGt = args.output_data_dir + "/gt/Resized_halforigional_" + baseImageName + \
#                                  '_darkness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
#                     imageio.imwrite(saveNameGt, nMask)
#         else:
#             # create brightness blur
#             for alpha in brightness_alpha:
#                 for beta in brightness_beta:
#                     bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
#                             alpha, beta)
#                     # Define brightness and darkness in image
#                     # maskdummy = np.squeeze(np.zeros(half_shape))
#                     # define_brightness_and_darkness_blur(bright_blurred_overlay_img, maskdummy)
#                     # cv2.imshow('BrightnessImage', dark_blur)
#                     # cv2.waitKey(0)
#                     # indicator for bright blur
#                     bright_mask = copy.deepcopy(saliencyMaskResized) * 255
#                     bright_mask[np.round(bright_mask, 4) > 0] = 255
#                     alpha_mask = bright_mask / 255.
#                     nMask = np.zeros(half_shape)
#                     define_brightness_and_darkness_blur(bright_blurred_overlay_img, nMask)
#                     nMask[nMask == 1] = 64
#                     nMask[nMask == 2] = 128
#                     nMask[nMask == 3] = 192
#                     nMask[nMask == 4] = 255
#                     final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
#                                                                 bright_blurred_overlay_img,alpha_mask)
#                     # cv2.imshow('DarknessImage', final_masked_blurred_image)
#                     # cv2.waitKey(0)
#                     # save final image
#                     saveName = args.output_data_dir + "/images/Resized_halforigional_" + baseImageName + \
#                                '_brightness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
#                     final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#                                      / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#                     # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#                     imageio.imwrite(saveName, final_masked_blurred_image)
#                     # create and save ground truth mask
#                     saveNameGt = args.output_data_dir + "/gt/Resized_halforigional_" + baseImageName + \
#                                  '_brightness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
#                     imageio.imwrite(saveNameGt, nMask)
#
#     count = 1
#     # now resize the images to small resolution
#     for ind in range(40):
#         i = index[ind]
#         baseImageName = images_list[i]
#         # trying to get 20 focus and 20 motion images even count
#         if 'focus' in baseImageName and count > 40:
#             continue
#         # https://idiotdeveloper.com/cv2-resize-resizing-image-using-opencv-python/
#         baseImageResized = cv2.resize(saliencyImages[i],(small_shape[1],small_shape[0]),interpolation=cv2.INTER_NEAREST)
#         #TODO make sure this doesnt affect truth labels
#         saliencyMaskResized = np.round(cv2.resize(saliencyMask[i],(small_shape[1],small_shape[0]),
#                                                   interpolation=cv2.INTER_NEAREST))[:,:,np.newaxis]
#
#         origionalMaskResized = np.round(cv2.resize(origionalSaliencyMask[i],(small_shape[1],small_shape[0]),
#                                                    interpolation=cv2.INTER_NEAREST))[:,:,np.newaxis]
#         if 'focus' in baseImageName:
#             for angle in motion_degree:
#                 for kernal_size in motion_kernal_size:
#                     # create motion blur
#                     motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(baseImageResized), kernal_size,
#                                                                        angle)
#                     # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
#                     # cv2.waitKey(0)
#                     # # Define brightness and darkness in image
#                     maskdummy = np.squeeze(np.zeros(small_shape))
#                     define_brightness_and_darkness_blur(motion_blurred_overlay_img, maskdummy)
#
#                     blur_mask = apply_motion_blur(copy.deepcopy(saliencyMaskResized) * 255, kernal_size, angle)
#                     motion_blur_mask = copy.deepcopy(blur_mask)
#                     motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
#                     alpha_mask = motion_blur_mask / 255.
#                     final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
#                                                                     motion_blurred_overlay_img,
#                                                                     alpha_mask[:, :, np.newaxis])
#                     # cv2.imshow('MotionImage', final_masked_blurred_image)
#                     # cv2.waitKey(0)
#                     nMask = copy.deepcopy(origionalMaskResized)
#                     #define_brightness_and_darkness_blur(saliencyImages[i], nMask)
#                     nMask[motion_blur_mask == 255] = 1
#                     maskdummy[np.squeeze(motion_blur_mask) != 255] = 255
#                     nMask[maskdummy == 3] = 3
#                     nMask[maskdummy == 4] = 4
#                     nMask[nMask == 1] = 64
#                     nMask[nMask == 2] = 128
#                     nMask[nMask == 3] = 192
#                     nMask[nMask == 4] = 255
#                     saveName = args.output_data_dir + "/images/Resized_small_" + baseImageName + \
#                                '_motion_angle_' + str(angle) + "_kernal_size_" + str(kernal_size) + args.data_extension
#                     final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#                                      / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#                     # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#                     imageio.imwrite(saveName, final_masked_blurred_image)
#                     # create and save ground truth mask
#                     saveNameGt = args.output_data_dir + "/gt/Resized_small_" + baseImageName + \
#                                  '_motion_angle_' + str(angle) + "_kernal_size_" + str(kernal_size) + \
#                                  args.data_extension
#                     cv2.imwrite(saveNameGt, nMask)
#         else:
#             # create out of focus blur
#             for kernal_size in focus_kernal_size:
#                 focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(baseImageResized), kernal_size)
#
#                 # # Define brightness and darkness in image
#                 maskdummy = np.squeeze(np.zeros(small_shape))
#                 define_brightness_and_darkness_blur(focus_blurred_overlay_img, maskdummy)
#
#                 # now make the mask
#                 blur_mask = create_out_of_focus_blur(copy.deepcopy(saliencyMaskResized) * 255., kernal_size)
#                 focus_blur_mask = copy.deepcopy(blur_mask)
#                 focus_blur_mask[np.round(blur_mask / 255., 1) > 0] = 255
#                 alpha_mask = focus_blur_mask / 255.
#                 final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),focus_blurred_overlay_img,
#                                                                 alpha_mask)
#                 nMask = copy.deepcopy(origionalMaskResized)
#                 #define_brightness_and_darkness_blur(saliencyImages[i], nMask)
#                 nMask[focus_blur_mask == 255] = 2
#                 maskdummy[np.squeeze(focus_blur_mask) != 255] = 255
#                 nMask[maskdummy == 3] = 3
#                 nMask[maskdummy == 4] = 4
#                 nMask[nMask == 1] = 64
#                 nMask[nMask == 2] = 128
#                 nMask[nMask == 3] = 192
#                 nMask[nMask == 4] = 255
#                 saveName = args.output_data_dir + "/images/Resized_small_" + baseImageName + \
#                            '_focus_kernal_size_' + str(kernal_size) + args.data_extension
#                 final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#                 # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#                 imageio.imwrite(saveName, final_masked_blurred_image)
#                 # create and save ground truth mask
#                 saveNameGt = args.output_data_dir + "/gt/Resized_small_" + baseImageName + \
#                              '_focus_kernal_size_' + str(kernal_size) + args.data_extension
#                 cv2.imwrite(saveNameGt, nMask)
#                 count += 1
#         if i % 2 == 0:
#             # create darkness blur
#             for alpha in darkness_alpha:
#                 for beta in darkness_beta:
#                     dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
#                                                                                        alpha, beta)
#                     # Define brightness and darkness in image
#                     # maskdummy = np.squeeze(np.zeros(small_shape))
#                     # define_brightness_and_darkness_blur(dark_blurred_overlay_img, maskdummy)
#                     # cv2.imshow('BrightnessImage', dark_blur)
#                     # cv2.waitKey(0)
#                     # indicator for dark blur
#                     dark_mask = copy.deepcopy(saliencyMaskResized) * 255
#                     dark_mask[np.round(dark_mask, 4) > 0] = 255
#                     alpha_mask = dark_mask / 255.
#                     nMask = np.zeros(small_shape)
#                     define_brightness_and_darkness_blur(dark_blurred_overlay_img, nMask)
#                     nMask[nMask == 1] = 64
#                     nMask[nMask == 2] = 128
#                     nMask[nMask == 3] = 192
#                     nMask[nMask == 4] = 255
#                     final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),dark_blurred_overlay_img,
#                                                                     alpha_mask)
#                     # cv2.imshow('DarknessImage', final_masked_blurred_image)
#                     # cv2.waitKey(0)
#                     # save just the darkness
#                     # save final image
#                     saveName = args.output_data_dir + "/images/Resized_small_" + baseImageName + \
#                                '_darkness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
#                     final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#                                      / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#                     # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#                     imageio.imwrite(saveName, final_masked_blurred_image)
#                     # create and save ground truth mask
#                     saveNameGt = args.output_data_dir + "/gt/Resized_small_" + baseImageName + \
#                                  '_darkness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
#                     cv2.imwrite(saveNameGt, nMask)
#         else:
#             # create brightness blur
#             for alpha in brightness_alpha:
#                 for beta in brightness_beta:
#                     bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
#                             alpha, beta)
#                     # Define brightness and darkness in image
#                     # maskdummy = np.squeeze(np.zeros(small_shape))
#                     # define_brightness_and_darkness_blur(bright_blurred_overlay_img, maskdummy)
#                     # cv2.imshow('BrightnessImage', dark_blur)
#                     # cv2.waitKey(0)
#                     # indicator for bright blur
#                     bright_mask = copy.deepcopy(saliencyMaskResized) * 255
#                     bright_mask[np.round(bright_mask, 4) > 0] = 255
#                     alpha_mask = bright_mask / 255.
#                     nMask = np.zeros(small_shape)
#                     define_brightness_and_darkness_blur(bright_blurred_overlay_img, nMask)
#                     nMask[nMask == 1] = 64
#                     nMask[nMask == 2] = 128
#                     nMask[nMask == 3] = 192
#                     nMask[nMask == 4] = 255
#                     final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
#                                                                 bright_blurred_overlay_img,alpha_mask)
#                     # cv2.imshow('DarknessImage', final_masked_blurred_image)
#                     # cv2.waitKey(0)
#                     # save final image
#                     saveName = args.output_data_dir + "/images/Resized_small_" + baseImageName + \
#                                '_brightness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
#                     final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
#                                      / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#                     # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
#                     imageio.imwrite(saveName, final_masked_blurred_image)
#                     # create and save ground truth mask
#                     saveNameGt = args.output_data_dir + "/gt/Resized_small_" + baseImageName + \
#                                  '_brightness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
#                     cv2.imwrite(saveNameGt, nMask)

def create_cuhk_dataset_for_sensitivity(args):
    # angles range for dataset
    motion_degree = np.array([30, 60, 90, 120, 150, 180])
    motion_kernal_size = np.array([3,5,7,9,25,33,47])
    focus_kernal_size = np.array([3,5,7,9,25,33,47])
    darkness_alpha = np.array([0.1,0.4,0.6,0.9])
    darkness_beta = np.array([0,-50,-100,-200])
    brightness_alpha = np.array([1.1,1.4,1.6,1.9])
    brightness_beta = np.array([0,50,100,200])
    # all ranges of motion, focus, darkness and brightness images are created
    final_shape = (480, 640)
    # make it with respect to aspect ratio
    scale = 60
    width = int(480 * scale / 100)
    height = int(640 * scale / 100)
    percent60_shape = (width, height)
    scale = 30
    width = int(480 * scale / 100)
    height = int(640 * scale / 100)
    percent30_shape = (width, height)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir + '/image', regx='/*.(jpg|JPG)', printable=False))
    images_gt_list = sorted(tl.files.load_file_list(path=args.data_dir + '/gt', regx='/*.(png|PNG)', printable=False))
    # these images have out of focus blur already
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir + '/image/', n_threads=100, mode='RGB')
    gtOrigonal = read_all_imgs(images_gt_list, path=args.data_dir + '/gt/', n_threads=100, mode='RGB2GRAY2')
    saliencyImages = copy.deepcopy(imagesOrigonal)
    saliencyMask = copy.deepcopy(gtOrigonal)
    origionalSaliencyMask = copy.deepcopy(saliencyMask)

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
        gtMask = np.ones((final_shape[0], final_shape[1], 1))
        gtMask[y1:y2, x1:x2] = saliencyMask[i][y1o:y2o, x1o:x2o]
        saliencyImages[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        new_mask = np.zeros(gtMask.shape)
        new_mask[gtMask == 0] = 0
        new_mask[gtMask == 1] = 3
        images_list[i] = images_list[i].split('.')[0]
        if 'motion' in images_list[i]:
            new_mask[gtMask == 255] = 1
        else:
            new_mask[gtMask == 255] = 2
        mask = np.zeros(new_mask.shape)
        mask[new_mask > 0] = 1
        saliencyMask[i] = np.logical_not(mask) * 1.0
        origionalSaliencyMask[i] = new_mask

    # now we will go through all of the images and make the dataset
    index = np.arange(start=0,stop=len(saliencyImages),step=1)
    np.random.shuffle(index)
    done = False
    ind = 0
    countf = 0
    countm = 0
    while ~done:
        i = index[ind]
        baseImageName = images_list[i]
        ind += 1
        if countf == 30 and countm == 30:
            done = True
            break
        # trying to get 20 focus and 20 motion images even count
        if 'focus' in baseImageName and countm == 30:
            continue
        if 'motion' in baseImageName and countf == 30:
            continue
        if 'focus' in baseImageName:
            for angle in motion_degree:
                for kernal_size in motion_kernal_size:
                    # create motion blur
                    motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(saliencyImages[i]), kernal_size,
                                                                       angle)
                    blur_mask = apply_motion_blur(copy.deepcopy(saliencyMask[i]) * 255, kernal_size, angle)
                    motion_blur_mask = copy.deepcopy(blur_mask)
                    motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                    alpha_mask = motion_blur_mask / 255.
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(saliencyImages[i]),
                                                                    motion_blurred_overlay_img,
                                                                    alpha_mask[:, :, np.newaxis])
                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    nMask = copy.deepcopy(origionalSaliencyMask[i])
                    #define_brightness_and_darkness_blur(saliencyImages[i], nMask)
                    nMask[motion_blur_mask == 255] = 1
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    saveName = args.output_data_dir + "/images/" + baseImageName + '_motion_angle_' + str(angle) + \
                                   "_kernal_size_" + str(kernal_size) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_motion_angle_' + str(angle) + \
                                     "_kernal_size_" + str(kernal_size) + args.data_extension
                    cv2.imwrite(saveNameGt, nMask.astype(np.uint8))
            countm += 1
        else:
            # create out of focus blur
            for kernal_size in focus_kernal_size:
                focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(saliencyImages[i]), kernal_size)

                # now make the mask
                blur_mask = create_out_of_focus_blur(copy.deepcopy(saliencyMask[i]) * 255., kernal_size)
                focus_blur_mask = copy.deepcopy(blur_mask)
                focus_blur_mask[np.round(blur_mask / 255., 1) > 0] = 255
                alpha_mask = focus_blur_mask / 255.
                final_masked_blurred_image = alpha_blending(copy.deepcopy(saliencyImages[i]),focus_blurred_overlay_img,
                                                                alpha_mask)
                nMask = copy.deepcopy(origionalSaliencyMask[i])
                #define_brightness_and_darkness_blur(saliencyImages[i], nMask)
                nMask[focus_blur_mask == 255] = 2
                nMask[nMask == 1] = 64
                nMask[nMask == 2] = 128
                nMask[nMask == 3] = 192
                nMask[nMask == 4] = 255
                saveName = args.output_data_dir + "/images/" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
                               + args.data_extension
                final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                 / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                imageio.imwrite(saveName, final_masked_blurred_image)
                # create and save ground truth mask
                saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
                                 + args.data_extension
                cv2.imwrite(saveNameGt, nMask.astype(np.uint8))
            countf += 1
        if i % 2 == 0:
            # create darkness blur
            for alpha in darkness_alpha:
                for beta in darkness_beta:
                    dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[i]),
                                                                                       alpha, beta)
                    # cv2.waitKey(0)
                    # indicator for dark blur
                    dark_mask = copy.deepcopy(saliencyMask[i]) * 255
                    dark_mask[np.round(dark_mask, 4) > 0] = 255
                    alpha_mask = dark_mask / 255.
                    nMask = copy.deepcopy(origionalSaliencyMask[i])#np.zeros(final_shape)
                    nMask[np.squeeze(dark_mask) == 255] = 3
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(saliencyImages[i]),dark_blurred_overlay_img,
                                                                    alpha_mask)
                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # save just the darkness
                    # save final image
                    saveName = args.output_data_dir + "/images/" + baseImageName + '_darkness_alpha_' + str(alpha) + \
                                   "_beta_" + str(beta) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_darkness_alpha_' + str(alpha) + \
                                     "_beta_" + str(beta) + args.data_extension
                    cv2.imwrite(saveNameGt, nMask.astype(np.uint8))
        else:
            # create brightness blur
            for alpha in brightness_alpha:
                for beta in brightness_beta:
                    bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[i]),
                            alpha, beta)
                    # # Define brightness and darkness in image
                    # maskdummy = np.squeeze(np.zeros(final_shape))
                    # define_brightness_and_darkness_blur(bright_blurred_overlay_img, maskdummy)
                    # cv2.imshow('BrightnessImage', dark_blur)
                    # cv2.waitKey(0)
                    # indicator for bright blur
                    bright_mask = copy.deepcopy(saliencyMask[i]) * 255
                    bright_mask[np.round(bright_mask, 4) > 0] = 255
                    alpha_mask = bright_mask / 255.
                    nMask = copy.deepcopy(origionalSaliencyMask[i])#np.zeros(final_shape)
                    nMask[np.squeeze(bright_mask) == 255] = 4
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(saliencyImages[i]),bright_blurred_overlay_img,
                                                                    alpha_mask)
                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # save final image
                    saveName = args.output_data_dir + "/images/" + baseImageName + '_brightness_alpha_' + str(alpha) + \
                                   "_beta_" + str(beta) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_brightness_alpha_' + str(alpha) + \
                                     "_beta_" + str(beta) + args.data_extension
                    cv2.imwrite(saveNameGt, nMask.astype(np.uint8))

    #index = index[40:]
    np.random.shuffle(index)
    # now resize the images to half of the current resolution
    done = False
    ind = 0
    countf = 0
    countm = 0
    while ~done:
        i = index[ind]
        baseImageName = images_list[i]
        ind += 1
        if countf == 30 and countm == 30:
            done = True
            break
        # trying to get 20 focus and 20 motion images even count
        if 'focus' in baseImageName and countm == 30:
            continue
        if 'motion' in baseImageName and countf == 30:
            continue
        baseImageResized = cv2.resize(saliencyImages[i], (percent60_shape[1], percent60_shape[0]))
        saliencyMaskResized = np.round(cv2.resize(saliencyMask[i], (percent60_shape[1], percent60_shape[0]))[:,:,np.newaxis])
        origionalMaskResized = np.round(cv2.resize(origionalSaliencyMask[i], (percent60_shape[1], percent60_shape[0]))[:,:,np.newaxis])
        if 'focus' in baseImageName:
            for angle in motion_degree:
                for kernal_size in motion_kernal_size:
                    # create motion blur
                    motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(baseImageResized), kernal_size,
                                                                       angle)

                    blur_mask = apply_motion_blur(copy.deepcopy(saliencyMaskResized) * 255, kernal_size, angle)
                    motion_blur_mask = copy.deepcopy(blur_mask)
                    motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                    alpha_mask = motion_blur_mask / 255.
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
                                                                    motion_blurred_overlay_img,
                                                                    alpha_mask[:, :, np.newaxis])
                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    nMask = copy.deepcopy(origionalMaskResized)
                    nMask[motion_blur_mask == 255] = 1
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    saveName = args.output_data_dir + "/images/Resized_60percent_" + baseImageName + \
                               '_motion_angle_' + str(angle) + "_kernal_size_" + str(kernal_size) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/Resized_60percent_" + baseImageName + \
                                 '_motion_angle_' + str(angle) + "_kernal_size_" + str(kernal_size) + \
                                 args.data_extension
                    imageio.imwrite(saveNameGt, nMask.astype(np.uint8))
            countm += 1
        else:
            # create out of focus blur
            for kernal_size in focus_kernal_size:
                focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(baseImageResized), kernal_size)
                # now make the mask
                blur_mask = create_out_of_focus_blur(copy.deepcopy(saliencyMaskResized) * 255., kernal_size)
                focus_blur_mask = copy.deepcopy(blur_mask)
                focus_blur_mask[np.round(blur_mask / 255., 1) > 0] = 255
                alpha_mask = focus_blur_mask / 255.
                final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),focus_blurred_overlay_img,
                                                                alpha_mask)
                nMask = copy.deepcopy(origionalMaskResized)
                nMask[focus_blur_mask == 255] = 2
                nMask[nMask == 1] = 64
                nMask[nMask == 2] = 128
                nMask[nMask == 3] = 192
                nMask[nMask == 4] = 255
                saveName = args.output_data_dir + "/images/Resized_60percent_" + baseImageName + \
                           '_focus_kernal_size_' + str(kernal_size) + args.data_extension
                final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                 / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                imageio.imwrite(saveName, final_masked_blurred_image)
                # create and save ground truth mask
                saveNameGt = args.output_data_dir + "/gt/Resized_60percent_" + baseImageName + \
                             '_focus_kernal_size_' + str(kernal_size) + args.data_extension
                imageio.imwrite(saveNameGt, nMask.astype(np.uint8))
            countf += 1
        if i % 2 == 0:
            # create darkness blur
            for alpha in darkness_alpha:
                for beta in darkness_beta:
                    dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
                                                                                       alpha, beta)
                    # indicator for dark blur
                    dark_mask = copy.deepcopy(saliencyMaskResized) * 255
                    dark_mask[np.round(dark_mask, 4) > 0] = 255
                    alpha_mask = dark_mask / 255.
                    nMask = copy.deepcopy(origionalMaskResized)
                    nMask[np.squeeze(dark_mask) == 255] = 3
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
                                                                dark_blurred_overlay_img,alpha_mask)
                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # save just the darkness
                    # save final image
                    saveName = args.output_data_dir + "/images/Resized_60percent_" + baseImageName + \
                               '_darkness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/Resized_60percent_" + baseImageName + \
                                 '_darkness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
                    imageio.imwrite(saveNameGt, nMask.astype(np.uint8))
        else:
            # create brightness blur
            for alpha in brightness_alpha:
                for beta in brightness_beta:
                    bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
                            alpha, beta)
                    # Define brightness and darkness in image
                    # indicator for bright blur
                    bright_mask = copy.deepcopy(saliencyMaskResized) * 255
                    bright_mask[np.round(bright_mask, 4) > 0] = 255
                    alpha_mask = bright_mask / 255.
                    nMask = copy.deepcopy(origionalMaskResized)
                    nMask[np.squeeze(bright_mask) == 255] = 4
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
                                                                bright_blurred_overlay_img,alpha_mask)
                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # save final image
                    saveName = args.output_data_dir + "/images/Resized_60percent_" + baseImageName + \
                               '_brightness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/Resized_60percent_" + baseImageName + \
                                 '_brightness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
                    imageio.imwrite(saveNameGt, nMask.astype(np.uint8))

    #index = index[40:]
    np.random.shuffle(index)
    # now resize the images to small resolution
    done = False
    ind = 0
    countf = 0
    countm = 0
    while ~done:
        i = index[ind]
        baseImageName = images_list[i]
        ind += 1
        if countf == 30 and countm == 30:
            done = True
            break
        # trying to get 20 focus and 20 motion images even count
        if 'focus' in baseImageName and countm == 30:
            continue
        if 'motion' in baseImageName and countf == 30:
            continue
        # https://idiotdeveloper.com/cv2-resize-resizing-image-using-opencv-python/
        baseImageResized = cv2.resize(saliencyImages[i], (percent30_shape[1], percent30_shape[0]))
        saliencyMaskResized = np.round(cv2.resize(saliencyMask[i], (percent30_shape[1], percent30_shape[0])))[:,:,np.newaxis]

        origionalMaskResized = np.round(cv2.resize(origionalSaliencyMask[i], (percent30_shape[1], percent30_shape[0])))[:,:,np.newaxis]
        if 'focus' in baseImageName:
            for angle in motion_degree:
                for kernal_size in motion_kernal_size:
                    # create motion blur
                    motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(baseImageResized), kernal_size,
                                                                       angle)

                    blur_mask = apply_motion_blur(copy.deepcopy(saliencyMaskResized) * 255, kernal_size, angle)
                    motion_blur_mask = copy.deepcopy(blur_mask)
                    motion_blur_mask[np.round(blur_mask, 4) > 0] = 255
                    alpha_mask = motion_blur_mask / 255.
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
                                                                    motion_blurred_overlay_img,
                                                                    alpha_mask[:, :, np.newaxis])
                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    nMask = copy.deepcopy(origionalMaskResized)
                    nMask[motion_blur_mask == 255] = 1
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    saveName = args.output_data_dir + "/images/Resized_30percent_" + baseImageName + \
                               '_motion_angle_' + str(angle) + "_kernal_size_" + str(kernal_size) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/Resized_30percent_" + baseImageName + \
                                 '_motion_angle_' + str(angle) + "_kernal_size_" + str(kernal_size) + \
                                 args.data_extension
                    cv2.imwrite(saveNameGt, nMask.astype(np.uint8))
            countm +=1
        else:
            # create out of focus blur
            for kernal_size in focus_kernal_size:
                focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(baseImageResized), kernal_size)

                # now make the mask
                blur_mask = create_out_of_focus_blur(copy.deepcopy(saliencyMaskResized) * 255., kernal_size)
                focus_blur_mask = copy.deepcopy(blur_mask)
                focus_blur_mask[np.round(blur_mask / 255., 1) > 0] = 255
                alpha_mask = focus_blur_mask / 255.
                final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),focus_blurred_overlay_img,
                                                                alpha_mask)
                nMask = copy.deepcopy(origionalMaskResized)
                nMask[focus_blur_mask == 255] = 2
                nMask[nMask == 1] = 64
                nMask[nMask == 2] = 128
                nMask[nMask == 3] = 192
                nMask[nMask == 4] = 255
                saveName = args.output_data_dir + "/images/Resized_30percent_" + baseImageName + \
                           '_focus_kernal_size_' + str(kernal_size) + args.data_extension
                final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                 / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                imageio.imwrite(saveName, final_masked_blurred_image)
                # create and save ground truth mask
                saveNameGt = args.output_data_dir + "/gt/Resized_30percent_" + baseImageName + \
                             '_focus_kernal_size_' + str(kernal_size) + args.data_extension
                cv2.imwrite(saveNameGt, nMask.astype(np.uint8))
            countf += 1
        if i % 2 == 0:
            # create darkness blur
            for alpha in darkness_alpha:
                for beta in darkness_beta:
                    dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
                                                                                       alpha, beta)
                    # indicator for dark blur
                    dark_mask = copy.deepcopy(saliencyMaskResized) * 255
                    dark_mask[np.round(dark_mask, 4) > 0] = 255
                    alpha_mask = dark_mask / 255.
                    nMask = copy.deepcopy(origionalMaskResized)
                    nMask[np.squeeze(dark_mask) == 255] = 3
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),dark_blurred_overlay_img,
                                                                    alpha_mask)
                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # save just the darkness
                    # save final image
                    saveName = args.output_data_dir + "/images/Resized_30percent_" + baseImageName + \
                               '_darkness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/Resized_30percent_" + baseImageName + \
                                 '_darkness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
                    cv2.imwrite(saveNameGt, nMask.astype(np.uint8))
        else:
            # create brightness blur
            for alpha in brightness_alpha:
                for beta in brightness_beta:
                    bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
                            alpha, beta)
                    # indicator for bright blur
                    bright_mask = copy.deepcopy(saliencyMaskResized) * 255
                    bright_mask[np.round(bright_mask, 4) > 0] = 255
                    alpha_mask = bright_mask / 255.
                    nMask = copy.deepcopy(origionalMaskResized)
                    nMask[np.squeeze(bright_mask) == 255] = 4
                    nMask[nMask == 1] = 64
                    nMask[nMask == 2] = 128
                    nMask[nMask == 3] = 192
                    nMask[nMask == 4] = 255
                    final_masked_blurred_image = alpha_blending(copy.deepcopy(baseImageResized),
                                                                bright_blurred_overlay_img,alpha_mask)
                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                    # cv2.waitKey(0)
                    # save final image
                    saveName = args.output_data_dir + "/images/Resized_30percent_" + baseImageName + \
                               '_brightness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
                    final_masked_blurred_image = np.round(np.power((np.array(final_masked_blurred_image)[:, :, 0:3] * 1.)
                                     / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                    final_masked_blurred_image[np.isnan(final_masked_blurred_image)] = 0
                    # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                    imageio.imwrite(saveName, final_masked_blurred_image)
                    # create and save ground truth mask
                    saveNameGt = args.output_data_dir + "/gt/Resized_30percent_" + baseImageName + \
                                 '_brightness_alpha_' + str(alpha) + "_beta_" + str(beta) + args.data_extension
                    cv2.imwrite(saveNameGt, nMask.astype(np.uint8))

def create_muri_and_ssc_dataset_for_testing(args):
    #final_shape = (480, 640)
    final_shape = (224, 224)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir, regx='/*.(png|PNG)', printable=False))
    images_gt_list = sorted(tl.files.load_file_list(path=args.salinecy_data_dir, regx='/*.(png|PNG)', printable=False))
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir+'/', n_threads=100, mode='RGB')
    gtOrigonal = read_all_imgs(images_gt_list, path=args.salinecy_data_dir+'/', n_threads=100, mode='RGB2GRAY2')
    sailencyMask = []

    for i in range(len(imagesOrigonal)):
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
        if imagesOrigonal[i].shape[2] > 3:
            imagesOrigonal[i] = imagesOrigonal[i][:,:,0:3]
        new_image[y1:y2, x1:x2] = imagesOrigonal[i][y1o:y2o, x1o:x2o]
        # darkness blur deafult for 0 pixels
        gtMask = np.ones((final_shape[0], final_shape[1], 1))
        gtMask[y1:y2, x1:x2] = gtOrigonal[i][y1o:y2o, x1o:x2o]
        # gamma correction
        imagesOrigonal[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        images_list[i] = images_list[i].split('.')[0]
        sailencyMask.append(np.copy(gtMask))
        new_mask = np.zeros(gtMask.shape)
        # define brightness and darkness in images
        define_brightness_and_darkness_blur(imagesOrigonal[i], new_mask)
        gtOrigonal[i] = new_mask

    # save baseline images
    for i in range(len(imagesOrigonal)):
        # always remember gamma correction
        baseImage = np.round(np.power((imagesOrigonal[i] * 1.) / 255, (1 / gamma)) * 255.0).astype(np.uint8)
        baseImageName = images_list[i]
        saveName = args.output_data_dir + "/images/" + baseImageName + args.data_extension
        imageio.imsave(saveName, baseImage)
        # get ground truth mask
        saveName = args.output_data_dir + "/gt/" + baseImageName + args.data_extension
        nMask = np.zeros(final_shape)
        gt = np.squeeze(gtOrigonal[i])
        nMask[gt == 1] = 64
        nMask[gt == 2] = 128
        nMask[gt == 3] = 192
        nMask[gt == 4] = 255
        cv2.imwrite(saveName, nMask)

    # create test images of each blur type
    for i in range(len(imagesOrigonal)):
        baseImageName = images_list[i]
        for k in range(10): # random motion blur
            # motion blur
            kernal, angle = random_motion_blur_kernel()
            motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(imagesOrigonal[i]),kernal, angle)
            blur_mask = apply_motion_blur(copy.deepcopy(sailencyMask[i]) * 255, kernal, angle)
            nMask = np.zeros(blur_mask.shape)
            nMask[blur_mask > 0] = 1
            define_brightness_and_darkness_blur(motion_blurred_overlay_img, nMask)
            # save just the darkness
            # save final image
            saveName = args.output_data_dir + "/images/motion_" + baseImageName + '_' + str(k) + args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(motion_blurred_overlay_img)[:, :, 0:3] * 1.)
                                                  / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/motion_" + baseImageName + '_' + str(k) + args.data_extension
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveNameGt, nMask)
        ### out of focus blur
        for k in range(10):
            kernal = random_focus_blur_kernel()
            focus_blur_overlay_img = create_out_of_focus_blur(np.copy(imagesOrigonal[i]), kernal)
            blur_mask = create_out_of_focus_blur(copy.deepcopy(sailencyMask[i]) * 255, kernal)
            nMask = np.zeros(blur_mask.shape)
            nMask[blur_mask > 0] = 2
            define_brightness_and_darkness_blur(focus_blur_overlay_img, nMask)

            saveName = args.output_data_dir + "/images/focus_" + baseImageName + '_' + str(k) + args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(focus_blur_overlay_img)[:, :, 0:3] * 1.)
                                                           / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            imageio.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/focus_" + baseImageName + '_' + str(k) + args.data_extension
            # add focus identification to mask
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveName, nMask)
        # next we go to darkness blur
        for k in range(10):
            # create darkness blur
            alpha, beta = random_darkness_value()
            dark_blur = create_brightness_and_darkness_blur(np.copy(imagesOrigonal[i]), alpha, beta)
            nMask = np.zeros(final_shape)
            define_brightness_and_darkness_blur(dark_blur, nMask)
            final_masked_blurred_image = np.round(np.power((np.array(dark_blur)[:, :, 0:3] * 1.)
                                                           / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # save the image
            saveName = args.output_data_dir + "/images/darkness_" + baseImageName + '_' + str(k) + args.data_extension
            imageio.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/darkness_" + baseImageName + '_' + str(k) + args.data_extension
            # add darkness blur to mask image
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveName, nMask)
        # next we go to brightness blur
        for k in range(10):
            # create brightness blur
            alpha,beta = random_brightness_value()
            bright_blur = create_brightness_and_darkness_blur(np.copy(imagesOrigonal[i]), alpha, beta)
            nMask = np.zeros(final_shape)
            define_brightness_and_darkness_blur(bright_blur, nMask)
            final_masked_blurred_image = np.round(np.power((np.array(bright_blur)[:, :, 0:3] * 1.)
                                                           / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # save the image
            saveName = args.output_data_dir + "/images/brightness_" + baseImageName + '_' + str(k) + args.data_extension
            imageio.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/brightness_" + baseImageName + '_' + str(k) + args.data_extension
            # add brightness blur to mask image
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            cv2.imwrite(saveName, nMask)

def create_muri_and_scc_dataset_for_sensitivity(args):
    motion_degree = np.array([30, 60, 90, 120, 150, 180])
    motion_kernal_size = np.array([3, 5, 7, 9, 25, 33, 47])
    focus_kernal_size = np.array([3, 5, 7, 9, 25, 33, 47])
    darkness_alpha = np.array([0.1, 0.2, 0.3, 0.5, 0.7])
    darkness_beta = np.array([0, -50, -100, -200])
    brightness_alpha = np.array([1.5, 1.8, 2.0, 2.3, 2.5])
    brightness_beta = np.array([0, 50, 100, 200])
    # all ranges of motion, focus, darkness and brightness images are created
    final_shape = (224, 224)
    # half_shape = (int(480/2), int(640/2))
    # small_shape = (224,224)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_list = sorted(tl.files.load_file_list(path=args.data_dir + '/', regx='/*.(png|PNG)', printable=False))
    # these images have out of focus blur already
    imagesOrigonal = read_all_imgs(images_list, path=args.data_dir + '/', n_threads=100, mode='RGB')
    saliencyImages = copy.deepcopy(imagesOrigonal)

    for i in range(len(saliencyImages)):
        if saliencyImages[i].shape[0] > saliencyImages[i].shape[1]:
            saliencyImages[i] = cv2.rotate(saliencyImages[i], 0)
            # saliencyMask[i] = cv2.rotate(saliencyMask[i], 0)[:, :, np.newaxis]
        y1, y2 = max(0, int((final_shape[0] + 1 - saliencyImages[i].shape[0]) / 2)), \
                 min(saliencyImages[i].shape[0] + int((final_shape[0] + 1 - saliencyImages[i].shape[0]) / 2),
                     final_shape[0])
        x1, x2 = max(0, int((final_shape[1] + 1 - saliencyImages[i].shape[1]) / 2)), \
                 min(saliencyImages[i].shape[1] + int((final_shape[1] + 1 - saliencyImages[i].shape[1]) / 2),
                     final_shape[1])
        y1o, y2o = 0, min(saliencyImages[i].shape[0], final_shape[0])
        x1o, x2o = 0, min(saliencyImages[i].shape[1], final_shape[1])

        new_image = np.zeros((final_shape[0], final_shape[1], 3))
        if saliencyImages[i].shape[2] > 3:
            saliencyImages[i] = saliencyImages[i][:,:,0:3]
        new_image[y1:y2, x1:x2] = saliencyImages[i][y1o:y2o, x1o:x2o]
        # # darkness blur deafult for 0 pixels
        # gtMask = np.ones((final_shape[0], final_shape[1], 1))
        # gtMask[y1:y2, x1:x2] = saliencyMask[i][y1o:y2o, x1o:x2o]
        # gamma correction
        saliencyImages[i] = 255.0 * np.power((new_image * 1.) / 255.0, gamma)
        images_list[i] = images_list[i].split('.')[0]
        #origionalSaliencyMask[i] = gtMask
        # new_mask = np.zeros(gtMask.shape)
        # define brightness and darkness in images
        # define_brightness_and_darkness_blur(saliencyImages[i], new_mask)
        # saliencyMask[i] = new_mask

    # now we will go through all of the images and make the dataset to make brightness and darkness blurs
    index = np.arange(0,len(saliencyImages),1)
    np.random.shuffle(index)
    # this already makes a huge dataset
    for i in range(len(saliencyImages)):
        baseImageName = images_list[i]
        for angle in motion_degree:
            for kernal_size in motion_kernal_size:
                # create motion blur
                motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(saliencyImages[i]), kernal_size,
                                                                   angle)
                nMask = np.ones(final_shape)
                define_brightness_and_darkness_blur(motion_blurred_overlay_img, nMask)
                nMask[nMask == 1] = 64
                nMask[nMask == 2] = 128
                nMask[nMask == 3] = 192
                nMask[nMask == 4] = 255
                saveName = args.output_data_dir + "/images/" + baseImageName + '_motion_angle_' + str(angle) + \
                               "_kernal_size_" + str(kernal_size) + args.data_extension
                final_masked_blurred_image = np.round(np.power((np.array(motion_blurred_overlay_img)[:, :, 0:3] * 1.)
                                 / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                imageio.imwrite(saveName, final_masked_blurred_image)
                # create and save ground truth mask
                saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_motion_angle_' + str(angle) + \
                                 "_kernal_size_" + str(kernal_size) + args.data_extension
                cv2.imwrite(saveNameGt, nMask)
        # create out of focus blur
        for kernal_size in focus_kernal_size:
            focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(saliencyImages[i]), kernal_size)
            nMask = np.ones(final_shape)*2
            define_brightness_and_darkness_blur(focus_blurred_overlay_img, nMask)
            nMask[nMask == 1] = 64
            nMask[nMask == 2] = 128
            nMask[nMask == 3] = 192
            nMask[nMask == 4] = 255
            saveName = args.output_data_dir + "/images/" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
                           + args.data_extension
            final_masked_blurred_image = np.round(np.power((np.array(focus_blurred_overlay_img)[:, :, 0:3] * 1.)
                             / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
            # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
            imageio.imwrite(saveName, final_masked_blurred_image)
            # create and save ground truth mask
            saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
                             + args.data_extension
            cv2.imwrite(saveNameGt, nMask)
        # create darkness blur
        for alpha in darkness_alpha:
            for beta in darkness_beta:
                dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[i]),
                                                                                   alpha, beta)
                # indicator for dark blur
                nMask = np.zeros(final_shape)
                define_brightness_and_darkness_blur(dark_blurred_overlay_img, nMask)
                nMask[nMask == 1] = 64
                nMask[nMask == 2] = 128
                nMask[nMask == 3] = 192
                nMask[nMask == 4] = 255
                # save just the darkness
                # save final image
                saveName = args.output_data_dir + "/images/" + baseImageName + '_darkness_alpha_' + str(alpha) + \
                               "_beta_" + str(beta) + args.data_extension
                final_masked_blurred_image = np.round(np.power((np.array(dark_blurred_overlay_img)[:, :, 0:3] * 1.)
                                 / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                imageio.imwrite(saveName, final_masked_blurred_image)
                # create and save ground truth mask
                saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_darkness_alpha_' + str(alpha) + \
                                 "_beta_" + str(beta) + args.data_extension
                cv2.imwrite(saveNameGt, nMask)
        # create brightness blur
        for alpha in brightness_alpha:
            for beta in brightness_beta:
                bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(saliencyImages[i]),
                        alpha, beta)
                nMask = np.zeros(final_shape)
                define_brightness_and_darkness_blur(bright_blurred_overlay_img, nMask)
                nMask[nMask == 1] = 64
                nMask[nMask == 2] = 128
                nMask[nMask == 3] = 192
                nMask[nMask == 4] = 255
                # save final image
                saveName = args.output_data_dir + "/images/" + baseImageName + '_brightness_alpha_' + str(alpha) + \
                               "_beta_" + str(beta) + args.data_extension
                final_masked_blurred_image = np.round(np.power((np.array(bright_blurred_overlay_img)[:, :, 0:3] * 1.)
                                 / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
                # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
                imageio.imwrite(saveName, final_masked_blurred_image)
                # create and save ground truth mask
                saveNameGt = args.output_data_dir + "/gt/" + baseImageName + '_brightness_alpha_' + str(alpha) + \
                                 "_beta_" + str(beta) + args.data_extension
                cv2.imwrite(saveNameGt, nMask)

    # now resize the images to half of the current resolution
    # for ind in range(3):
    #     i = index[ind]
    #     baseImageName = images_list[i]
    #     baseImageResized = cv2.resize(saliencyImages[i],(half_shape[1],half_shape[0]))
    #     for angle in motion_degree:
    #         for kernal_size in motion_kernal_size:
    #             # create motion blur
    #             motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(baseImageResized), kernal_size,
    #                                                            angle)
    #             nMask = np.ones(half_shape)
    #             define_brightness_and_darkness_blur(motion_blurred_overlay_img, nMask)
    #             nMask[nMask == 1] = 64
    #             nMask[nMask == 2] = 128
    #             nMask[nMask == 3] = 192
    #             nMask[nMask == 4] = 255
    #             saveName = args.output_data_dir + "/images/resized_half_" + baseImageName + '_motion_angle_' + str(angle) + \
    #                        "_kernal_size_" + str(kernal_size) + args.data_extension
    #             final_masked_blurred_image = np.round(np.power((np.array(motion_blurred_overlay_img)[:, :, 0:3] * 1.)
    #                                                            / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    #             # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
    #             imageio.imwrite(saveName, final_masked_blurred_image)
    #             # create and save ground truth mask
    #             saveNameGt = args.output_data_dir + "/gt/resized_half_" + baseImageName + '_motion_angle_' + str(angle) + \
    #                          "_kernal_size_" + str(kernal_size) + args.data_extension
    #             cv2.imwrite(saveNameGt, nMask)
    #     # create out of focus blur
    #     for kernal_size in focus_kernal_size:
    #         focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(baseImageResized), kernal_size)
    #
    #         # now make the mask
    #         nMask = np.ones(half_shape)*2
    #         define_brightness_and_darkness_blur(focus_blurred_overlay_img, nMask)
    #         nMask[nMask == 1] = 64
    #         nMask[nMask == 2] = 128
    #         nMask[nMask == 3] = 192
    #         nMask[nMask == 4] = 255
    #         saveName = args.output_data_dir + "/images/resized_half_" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
    #                    + args.data_extension
    #         final_masked_blurred_image = np.round(np.power((np.array(focus_blurred_overlay_img)[:, :, 0:3] * 1.)
    #                                                        / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    #         # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
    #         imageio.imwrite(saveName, final_masked_blurred_image)
    #         # create and save ground truth mask
    #         saveNameGt = args.output_data_dir + "/gt/resized_half_" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
    #                      + args.data_extension
    #         cv2.imwrite(saveNameGt, nMask)
    #     # create darkness blur
    #     for alpha in darkness_alpha:
    #         for beta in darkness_beta:
    #             dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
    #                                                                            alpha, beta)
    #             # indicator for dark blur
    #             nMask = np.zeros(half_shape)
    #             define_brightness_and_darkness_blur(dark_blurred_overlay_img, nMask)
    #             nMask[nMask == 1] = 64
    #             nMask[nMask == 2] = 128
    #             nMask[nMask == 3] = 192
    #             nMask[nMask == 4] = 255
    #             # save just the darkness
    #             # save final image
    #             saveName = args.output_data_dir + "/images/resized_half_" + baseImageName + '_darkness_alpha_' + str(alpha) + \
    #                        "_beta_" + str(beta) + args.data_extension
    #             final_masked_blurred_image = np.round(np.power((np.array(dark_blurred_overlay_img)[:, :, 0:3] * 1.)
    #                                                            / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    #             # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
    #             imageio.imwrite(saveName, final_masked_blurred_image)
    #             # create and save ground truth mask
    #             saveNameGt = args.output_data_dir + "/gt/resized_half_" + baseImageName + '_darkness_alpha_' + str(alpha) + \
    #                          "_beta_" + str(beta) + args.data_extension
    #             cv2.imwrite(saveNameGt, nMask)
    #     # create brightness blur
    #     for alpha in brightness_alpha:
    #         for beta in brightness_beta:
    #             bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
    #                                                                              alpha, beta)
    #             # indicator for bright blur
    #             nMask = np.zeros(half_shape)
    #             define_brightness_and_darkness_blur(bright_blurred_overlay_img, nMask)
    #             nMask[nMask == 1] = 64
    #             nMask[nMask == 2] = 128
    #             nMask[nMask == 3] = 192
    #             nMask[nMask == 4] = 255
    #             # save final image
    #             saveName = args.output_data_dir + "/images/resized_half_" + baseImageName + '_brightness_alpha_' + str(alpha) + \
    #                        "_beta_" + str(beta) + args.data_extension
    #             final_masked_blurred_image = np.round(np.power((np.array(bright_blurred_overlay_img)[:, :, 0:3] * 1.)
    #                                                            / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    #             # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
    #             imageio.imwrite(saveName, final_masked_blurred_image)
    #             # create and save ground truth mask
    #             saveNameGt = args.output_data_dir + "/gt/resized_half_" + baseImageName + '_brightness_alpha_' + str(alpha) + \
    #                          "_beta_" + str(beta) + args.data_extension
    #             cv2.imwrite(saveNameGt, nMask)
    #
    # # now resize the images to small resolution
    # for ind in range(3):
    #     i = index[ind]
    #     baseImageName = images_list[i]
    #     baseImageResized = cv2.resize(saliencyImages[i],small_shape)
    #     for angle in motion_degree:
    #         for kernal_size in motion_kernal_size:
    #             # create motion blur
    #             motion_blurred_overlay_img = apply_motion_blur(copy.deepcopy(baseImageResized), kernal_size,
    #                                                            angle)
    #             nMask = np.ones(small_shape)
    #             define_brightness_and_darkness_blur(motion_blurred_overlay_img, nMask)
    #             nMask[nMask == 1] = 64
    #             nMask[nMask == 2] = 128
    #             nMask[nMask == 3] = 192
    #             nMask[nMask == 4] = 255
    #             saveName = args.output_data_dir + "/images/resized_small_" + baseImageName + '_motion_angle_' + str(angle) + \
    #                        "_kernal_size_" + str(kernal_size) + args.data_extension
    #             final_masked_blurred_image = np.round(np.power((np.array(motion_blurred_overlay_img)[:, :, 0:3] * 1.)
    #                                                            / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    #             # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
    #             imageio.imwrite(saveName, final_masked_blurred_image)
    #             # create and save ground truth mask
    #             saveNameGt = args.output_data_dir + "/gt/resized_small_" + baseImageName + '_motion_angle_' + str(angle) + \
    #                          "_kernal_size_" + str(kernal_size) + args.data_extension
    #             cv2.imwrite(saveNameGt, nMask)
    #     # create out of focus blur
    #     for kernal_size in focus_kernal_size:
    #         focus_blurred_overlay_img = create_out_of_focus_blur(copy.deepcopy(baseImageResized), kernal_size)
    #
    #         # now make the mask
    #         nMask = np.ones(small_shape)*2
    #         define_brightness_and_darkness_blur(focus_blurred_overlay_img, nMask)
    #         nMask[nMask == 1] = 64
    #         nMask[nMask == 2] = 128
    #         nMask[nMask == 3] = 192
    #         nMask[nMask == 4] = 255
    #         saveName = args.output_data_dir + "/images/resized_small_" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
    #                    + args.data_extension
    #         final_masked_blurred_image = np.round(np.power((np.array(focus_blurred_overlay_img)[:, :, 0:3] * 1.)
    #                                                        / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    #         # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
    #         imageio.imwrite(saveName, final_masked_blurred_image)
    #         # create and save ground truth mask
    #         saveNameGt = args.output_data_dir + "/gt/resized_small_" + baseImageName + '_focus_kernal_size_' + str(kernal_size) \
    #                      + args.data_extension
    #         cv2.imwrite(saveNameGt, nMask)
    #     # create darkness blur
    #     for alpha in darkness_alpha:
    #         for beta in darkness_beta:
    #             dark_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
    #                                                                            alpha, beta)
    #             # indicator for dark blur
    #             nMask = np.zeros(small_shape)
    #             define_brightness_and_darkness_blur(dark_blurred_overlay_img, nMask)
    #             nMask[nMask == 1] = 64
    #             nMask[nMask == 2] = 128
    #             nMask[nMask == 3] = 192
    #             nMask[nMask == 4] = 255
    #             # save just the darkness
    #             # save final image
    #             saveName = args.output_data_dir + "/images/resized_small_" + baseImageName + '_darkness_alpha_' + str(alpha) + \
    #                        "_beta_" + str(beta) + args.data_extension
    #             final_masked_blurred_image = np.round(np.power((np.array(dark_blurred_overlay_img)[:, :, 0:3] * 1.)
    #                                                            / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    #             # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
    #             imageio.imwrite(saveName, final_masked_blurred_image)
    #             # create and save ground truth mask
    #             saveNameGt = args.output_data_dir + "/gt/resized_small_" + baseImageName + '_darkness_alpha_' + str(alpha) + \
    #                          "_beta_" + str(beta) + args.data_extension
    #             cv2.imwrite(saveNameGt, nMask)
    #     # create brightness blur
    #     for alpha in brightness_alpha:
    #         for beta in brightness_beta:
    #             bright_blurred_overlay_img = create_brightness_and_darkness_blur(copy.deepcopy(baseImageResized),
    #                                                                              alpha, beta)
    #             # indicator for bright blur
    #             nMask = np.zeros(small_shape)
    #             define_brightness_and_darkness_blur(bright_blurred_overlay_img, nMask)
    #             nMask[nMask == 1] = 64
    #             nMask[nMask == 2] = 128
    #             nMask[nMask == 3] = 192
    #             nMask[nMask == 4] = 255
    #             # save final image
    #             saveName = args.output_data_dir + "/images/resized_small_" + baseImageName + '_brightness_alpha_' + str(alpha) + \
    #                        "_beta_" + str(beta) + args.data_extension
    #             final_masked_blurred_image = np.round(np.power((np.array(bright_blurred_overlay_img)[:, :, 0:3] * 1.)
    #                                                            / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    #             # https: // note.nkmk.me / en / python - opencv - bgr - rgb - cvtcolor /
    #             imageio.imwrite(saveName, final_masked_blurred_image)
    #             # create and save ground truth mask
    #             saveNameGt = args.output_data_dir + "/gt/resized_small_" + baseImageName + '_brightness_alpha_' + str(alpha) + \
    #                          "_beta_" + str(beta) + args.data_extension
    #             cv2.imwrite(saveNameGt, nMask)

if __name__ == "__main__":
 # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS MURI CREATE BLUR DATASET')
    # directory data location
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_data_dir', type=str)
    parser.add_argument('--salinecy_data_dir', type=str,default=None)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str,default=".png")
    parser.add_argument('--is_testing', default=False, action='store_true')
    parser.add_argument('--is_cuhk_dataset', default=False, action='store_true')
    parser.add_argument('--is_muri_dataset', default=False, action='store_true')
    parser.add_argument('--is_sensitivity', default=False, action='store_true')
    args = parser.parse_args()
    if args.is_testing:
        if args.is_cuhk_dataset:
            create_cuhk_dataset_for_testing(args)
        elif args.is_muri_dataset:
            create_muri_and_ssc_dataset_for_testing(args)
    elif args.is_sensitivity:
        if args.is_cuhk_dataset:
            create_cuhk_dataset_for_sensitivity(args)
        elif args.is_muri_dataset:
            create_muri_and_scc_dataset_for_sensitivity(args)
    else:
        if args.is_cuhk_dataset:
            create_cuhk_dataset_for_training(args)
        # else:
        #     create_muri_dataset_for_training(args)


