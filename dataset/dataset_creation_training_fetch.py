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

# for my images
gamma = 1.5

# we want very extreme values for brightness and darkness since this is causing issues with discriminating
def random_darkness_value(amin=0.01, amax=0.6, bmin=-100, bmax=0):
    alpha = random.uniform(amin, amax)
    beta = random.uniform(bmin, bmax)
    return alpha, beta

def random_brightness_value(amin=1.5, amax=2.6, bmin=0, bmax=100):
    alpha = random.uniform(amin, amax)
    beta = random.uniform(bmin, bmax)
    return alpha, beta

def find_overexposure_and_underexposure(image,mask):
    imagegray = skimage.color.rgb2gray(image)
    mask[imagegray < 30] = 3
    mask[imagegray > 245] = 4
    return mask

def create_overexposure_and_underexposure_blur(image, alpha, beta):
    new_img = image * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img

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

# create training dataset type 1
# all real images overlayed for training
def create_dataset_for_training_type_1(args):
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
    images_sailency_exposed_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency_exposed',
                                                                  regx='/*.(png|PNG)',printable=False))
    images_sailency_focus_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency_focus',
                                                                  regx='/*.(png|PNG)',printable=False))
    images_sailency_normal_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency', regx='/*.(png|PNG)',
                                                        printable=False))
    # these images have out of focus blur already we need to create motion blur, and the over and under exposure images
    imagesNOrigonal = read_all_imgs(images_normal_list, path=args.data_dir +'/normal/', n_threads=100, mode='RGB')
    imagesFOrigonal = read_all_imgs(images_focus_list, path=args.data_dir +'/focus/', n_threads=100, mode='RGB')
    imagesMOrigonal = read_all_imgs(images_motion_list, path=args.data_dir + '/motion/', n_threads=100, mode='RGB')
    imagesOOrigonal = read_all_imgs(images_overexposure_list, path=args.data_dir + '/overexposure/', n_threads=100,
                                    mode='RGB')
    imagesUOrigonal = read_all_imgs(images_underexposure_list, path=args.data_dir + '/underexposure/',
                                    n_threads=100, mode='RGB')
    imagesNSaliency = read_all_imgs(images_sailency_normal_list, path=args.data_dir + '/saliency/',
                                    n_threads=100, mode='GRAY')
    imagesESaliency = read_all_imgs(images_sailency_exposed_list, path=args.data_dir + '/saliency_exposed/',
                                    n_threads=100, mode='GRAY')
    imagesFSaliency = read_all_imgs(images_sailency_focus_list, path=args.data_dir + '/saliency_focus/',
                                    n_threads=100,mode='GRAY')

    # save normal images
    # 1st get a random motion image
    index = np.arange(0,len(imagesMOrigonal),1)
    for i in range(400):
        idx_motion = random.choice(index)
        image_base = 255.0 * np.power((copy.deepcopy(imagesMOrigonal[idx_motion]) * 1.) / 255.0, gamma)
        nMask = np.ones((final_shape[0], final_shape[1]))
        nMask = find_overexposure_and_underexposure(np.power((image_base * 1.)/ 255.0, (1.0 / gamma)) * 255.0,nMask)

        # overlay normal image
        idx_normal = random.choice(index)
        image_normal = 255.0 * np.power((copy.deepcopy(imagesNOrigonal[idx_normal])* 1.) / 255.0, gamma)
        image_saliency_normal = np.squeeze(copy.deepcopy(imagesNSaliency[idx_normal]))
        # now overlay the images
        image_saliency_normal[image_saliency_normal > 0.1] = 255
        alpha = image_saliency_normal / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(image_base, image_normal, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_normal[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_normal[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 0
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # overlay focus image
        idx_focus = random.choice(index)
        image_focus = 255.0 * np.power((copy.deepcopy(imagesFOrigonal[idx_focus]) * 1.) / 255.0, gamma)
        image_saliency_focus = np.squeeze(copy.deepcopy(imagesFSaliency[idx_focus]))
        # now overlay the images
        image_saliency_focus[image_saliency_focus > 0.1] = 255
        alpha = image_saliency_focus / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                             image_focus, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_focus[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_focus[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 2
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # overlay overexposure image
        idx_overexposure = random.choice(index)
        image_overexposure = 255.0 * np.power((copy.deepcopy(imagesOOrigonal[idx_overexposure]) * 1.) / 255.0, gamma)
        image_saliency_overexposure = np.squeeze(copy.deepcopy(imagesESaliency[idx_overexposure]))
        # now overlay the images
        image_saliency_overexposure[image_saliency_overexposure > 0.1] = 255
        alpha = image_saliency_overexposure / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                             image_overexposure, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_overexposure[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_overexposure[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 4
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # overlay underexposure image
        idx_underexposure = random.choice(index)
        image_underexposure = 255.0 * np.power((copy.deepcopy(imagesUOrigonal[idx_underexposure]) * 1.) / 255.0, gamma)
        image_saliency_underexposure = np.squeeze(copy.deepcopy(imagesESaliency[idx_underexposure]))
        # now overlay the images
        image_saliency_underexposure[image_saliency_underexposure > 0.1] = 255
        alpha = image_saliency_underexposure / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                             image_underexposure, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_underexposure[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_underexposure[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 3
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # save image
        saveName = args.output_data_dir + "/images/"+ str(i) + "_" + args.data_extension
        final_masked_blurred_image = np.round(np.power((final_masked_blurred_image * 1.)
                                              / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
        imageio.imsave(saveName, final_masked_blurred_image)
        # save gt
        saveName = args.output_data_dir + "/gt/"+str(i) + "_" + args.data_extension
        nMask[nMask == 1] = 64
        nMask[nMask == 2] = 128
        nMask[nMask == 3] = 192
        nMask[nMask == 4] = 255
        cv2.imwrite(saveName, nMask)

# create training dataset type 2
# real motion, normal and focus images overlayed with fake brightness and darkness images from normal images
def create_dataset_for_training_type_2(args):
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
    images_sailency_focus_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency_focus',
                                                                  regx='/*.(png|PNG)',printable=False))
    images_sailency_normal_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency', regx='/*.(png|PNG)',
                                                        printable=False))
    # these images have out of focus blur already we need to create motion blur, and the over and under exposure images
    imagesNOrigonal = read_all_imgs(images_normal_list, path=args.data_dir +'/normal/', n_threads=100, mode='RGB')
    imagesFOrigonal = read_all_imgs(images_focus_list, path=args.data_dir +'/focus/', n_threads=100, mode='RGB')
    imagesMOrigonal = read_all_imgs(images_motion_list, path=args.data_dir + '/motion/', n_threads=100, mode='RGB')
    imagesNSaliency = read_all_imgs(images_sailency_normal_list, path=args.data_dir + '/saliency/',
                                    n_threads=100, mode='GRAY')
    imagesFSaliency = read_all_imgs(images_sailency_focus_list, path=args.data_dir + '/saliency_focus/',
                                    n_threads=100,mode='GRAY')

    # save normal images
    # 1st get a random motion image
    index = np.arange(0,len(imagesMOrigonal),1)
    for i in range(400):
        idx_motion = random.choice(index)
        image_base = 255.0 * np.power((copy.deepcopy(imagesMOrigonal[idx_motion]) * 1.) / 255.0, gamma)
        nMask = np.ones((final_shape[0], final_shape[1]))
        nMask = find_overexposure_and_underexposure(np.power((image_base * 1.)/ 255.0, (1.0 / gamma)) * 255.0,nMask)

        # overlay normal image
        idx_normal = random.choice(index)
        image_normal = 255.0 * np.power((copy.deepcopy(imagesNOrigonal[idx_normal])* 1.) / 255.0, gamma)
        image_saliency_normal = np.squeeze(copy.deepcopy(imagesNSaliency[idx_normal]))
        # now overlay the images
        image_saliency_normal[image_saliency_normal > 0.1] = 255
        alpha = image_saliency_normal / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(image_base, image_normal, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_normal[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_normal[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 0
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # overlay focus image
        idx_focus = random.choice(index)
        image_focus = 255.0 * np.power((copy.deepcopy(imagesFOrigonal[idx_focus]) * 1.) / 255.0, gamma)
        image_saliency_focus = np.squeeze(copy.deepcopy(imagesFSaliency[idx_focus]))
        # now overlay the images
        image_saliency_focus[image_saliency_focus > 0.1] = 255
        alpha = image_saliency_focus / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                             image_focus, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_focus[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_focus[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 2
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # overlay overexposure image
        idx_overexposure = random.choice(index)
        image_overexposure = 255.0 * np.power((copy.deepcopy(imagesNOrigonal[idx_overexposure])* 1.) / 255.0, gamma)
        a, b = random_brightness_value()
        image_overexposure = create_overexposure_and_underexposure_blur(image_overexposure,a,b)
        image_saliency_overexposure = np.squeeze(copy.deepcopy(imagesNSaliency[idx_overexposure]))
        # now overlay the images
        image_saliency_overexposure[image_saliency_overexposure > 0.1] = 255
        alpha = image_saliency_overexposure / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                             image_overexposure, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_overexposure[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_overexposure[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 4
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # overlay underexposure image
        idx_underexposure = random.choice(index)
        image_underexposure = 255.0 * np.power((copy.deepcopy(imagesNOrigonal[idx_underexposure])* 1.) / 255.0, gamma)
        a, b = random_darkness_value()
        image_underexposure = create_overexposure_and_underexposure_blur(image_underexposure,a,b)
        image_saliency_underexposure = np.squeeze(copy.deepcopy(imagesNSaliency[idx_underexposure]))
        # now overlay the images
        image_saliency_underexposure[image_saliency_underexposure > 0.1] = 255
        alpha = image_saliency_underexposure / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                             image_underexposure, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_underexposure[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_underexposure[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 3
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # save image
        saveName = args.output_data_dir + "/images/"+ str(i) + "_" + args.data_extension
        final_masked_blurred_image = np.round(np.power((final_masked_blurred_image * 1.)
                                              / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
        imageio.imsave(saveName, final_masked_blurred_image)
        # save gt
        saveName = args.output_data_dir + "/gt/"+str(i) + "_" + args.data_extension
        nMask[nMask == 1] = 64
        nMask[nMask == 2] = 128
        nMask[nMask == 3] = 192
        nMask[nMask == 4] = 255
        cv2.imwrite(saveName, nMask)

# create training dataset type 2
# real motion, normal and focus images overlayed with fake and real brightness and darkness images from normal images
def create_dataset_for_training_type_3(args):
    final_shape = (480, 640)
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    images_normal_list = sorted(tl.files.load_file_list(path=args.data_dir + '/normal', regx='/*.(png|PNG)',
                                                        printable=False))
    images_focus_list = sorted(tl.files.load_file_list(path=args.data_dir + '/focus', regx='/*.(png|PNG)',
                                                       printable=False))
    images_motion_list = sorted(tl.files.load_file_list(path=args.data_dir + '/motion', regx='/*.(png|PNG)',
                                                        printable=False))
    images_overexposure_list = sorted(tl.files.load_file_list(path=args.data_dir + '/overexposure', regx='/*.(png|PNG)',
                                                              printable=False))
    images_underexposure_list = sorted(tl.files.load_file_list(path=args.data_dir + '/underexposure',
                                                               regx='/*.(png|PNG)', printable=False))
    images_sailency_exposed_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency_exposed',
                                                                  regx='/*.(png|PNG)', printable=False))
    images_sailency_focus_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency_focus',
                                                                regx='/*.(png|PNG)', printable=False))
    images_sailency_normal_list = sorted(tl.files.load_file_list(path=args.data_dir + '/saliency', regx='/*.(png|PNG)',
                                                                 printable=False))
    # these images have out of focus blur already we need to create motion blur, and the over and under exposure images
    imagesNOrigonal = read_all_imgs(images_normal_list, path=args.data_dir + '/normal/', n_threads=100, mode='RGB')
    imagesFOrigonal = read_all_imgs(images_focus_list, path=args.data_dir + '/focus/', n_threads=100, mode='RGB')
    imagesMOrigonal = read_all_imgs(images_motion_list, path=args.data_dir + '/motion/', n_threads=100, mode='RGB')
    imagesOOrigonal = read_all_imgs(images_overexposure_list, path=args.data_dir + '/overexposure/', n_threads=100,
                                    mode='RGB')
    imagesUOrigonal = read_all_imgs(images_underexposure_list, path=args.data_dir + '/underexposure/',
                                    n_threads=100, mode='RGB')
    imagesNSaliency = read_all_imgs(images_sailency_normal_list, path=args.data_dir + '/saliency/',
                                    n_threads=100, mode='GRAY')
    imagesESaliency = read_all_imgs(images_sailency_exposed_list, path=args.data_dir + '/saliency_exposed/',
                                    n_threads=100, mode='GRAY')
    imagesFSaliency = read_all_imgs(images_sailency_focus_list, path=args.data_dir + '/saliency_focus/',
                                    n_threads=100, mode='GRAY')

    # save normal images
    # 1st get a random motion image
    index = np.arange(0,len(imagesMOrigonal),1)
    r_or_f = [0,1]
    for i in range(400):
        idx_motion = random.choice(index)
        image_base = 255.0 * np.power((copy.deepcopy(imagesMOrigonal[idx_motion]) * 1.) / 255.0, gamma)
        nMask = np.ones((final_shape[0], final_shape[1]))
        nMask = find_overexposure_and_underexposure(np.power((image_base * 1.)/ 255.0, (1.0 / gamma)) * 255.0,nMask)

        # overlay normal image
        idx_normal = random.choice(index)
        image_normal = 255.0 * np.power((copy.deepcopy(imagesNOrigonal[idx_normal])* 1.) / 255.0, gamma)
        image_saliency_normal = np.squeeze(copy.deepcopy(imagesNSaliency[idx_normal]))
        # now overlay the images
        image_saliency_normal[image_saliency_normal > 0.1] = 255
        alpha = image_saliency_normal / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(image_base, image_normal, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_normal[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_normal[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 0
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # overlay focus image
        idx_focus = random.choice(index)
        image_focus = 255.0 * np.power((copy.deepcopy(imagesFOrigonal[idx_focus]) * 1.) / 255.0, gamma)
        image_saliency_focus = np.squeeze(copy.deepcopy(imagesFSaliency[idx_focus]))
        # now overlay the images
        image_saliency_focus[image_saliency_focus > 0.1] = 255
        alpha = image_saliency_focus / 255.
        placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                     np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
        final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                             image_focus, placement[1],
                                                                             placement[0], alpha)
        nMask[max(0, placement[0]) + np.argwhere(image_saliency_focus[y1o:y2o, x1o:x2o] == 255)[:, 0],
              max(0, placement[1]) + np.argwhere(image_saliency_focus[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 2
        nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                             (1.0 / gamma)) * 255.0, nMask)

        # overlay overexposure image
        # 1 fake 0 real
        if np.random.choice(r_or_f) == 1:
            idx_overexposure = random.choice(index)
            image_overexposure = 255.0 * np.power((copy.deepcopy(imagesNOrigonal[idx_overexposure])* 1.) / 255.0, gamma)
            a, b = random_brightness_value()
            image_overexposure = create_overexposure_and_underexposure_blur(image_overexposure,a,b)
            image_saliency_overexposure = np.squeeze(copy.deepcopy(imagesNSaliency[idx_overexposure]))
            # now overlay the images
            image_saliency_overexposure[image_saliency_overexposure > 0.1] = 255
            alpha = image_saliency_overexposure / 255.
            placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                         np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
            final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                                 image_overexposure, placement[1],
                                                                                 placement[0], alpha)
            nMask[max(0, placement[0]) + np.argwhere(image_saliency_overexposure[y1o:y2o, x1o:x2o] == 255)[:, 0],
                  max(0, placement[1]) + np.argwhere(image_saliency_overexposure[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 4
            nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                                 (1.0 / gamma)) * 255.0, nMask)
        else:
            idx_overexposure = random.choice(index)
            image_overexposure = 255.0 * np.power((copy.deepcopy(imagesOOrigonal[idx_overexposure]) * 1.) / 255.0,
                                                  gamma)
            image_saliency_overexposure = np.squeeze(copy.deepcopy(imagesESaliency[idx_overexposure]))
            # now overlay the images
            image_saliency_overexposure[image_saliency_overexposure > 0.1] = 255
            alpha = image_saliency_overexposure / 255.
            placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                         np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
            final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                                 image_overexposure, placement[1],
                                                                                 placement[0], alpha)
            nMask[max(0, placement[0]) + np.argwhere(image_saliency_overexposure[y1o:y2o, x1o:x2o] == 255)[:, 0],
                  max(0, placement[1]) + np.argwhere(image_saliency_overexposure[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 4
            nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.) / 255.0,
                                                                 (1.0 / gamma)) * 255.0, nMask)

        # overlay underexposure image
        # 1 fake 0 real
        if np.random.choice(r_or_f) == 1:
            idx_underexposure = random.choice(index)
            image_underexposure = 255.0 * np.power((copy.deepcopy(imagesNOrigonal[idx_underexposure])* 1.) / 255.0, gamma)
            a, b = random_darkness_value()
            image_underexposure = create_overexposure_and_underexposure_blur(image_underexposure,a,b)
            image_saliency_underexposure = np.squeeze(copy.deepcopy(imagesNSaliency[idx_underexposure]))
            # now overlay the images
            image_saliency_underexposure[image_saliency_underexposure > 0.1] = 255
            alpha = image_saliency_underexposure / 255.
            placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                         np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
            final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                                 image_underexposure, placement[1],
                                                                                 placement[0], alpha)
            nMask[max(0, placement[0]) + np.argwhere(image_saliency_underexposure[y1o:y2o, x1o:x2o] == 255)[:, 0],
                  max(0, placement[1]) + np.argwhere(image_saliency_underexposure[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 3
            nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.)/ 255.0,
                                                                 (1.0 / gamma)) * 255.0, nMask)
        else:
            idx_underexposure = random.choice(index)
            image_underexposure = 255.0 * np.power((copy.deepcopy(imagesUOrigonal[idx_underexposure]) * 1.) / 255.0,
                                                   gamma)
            image_saliency_underexposure = np.squeeze(copy.deepcopy(imagesESaliency[idx_underexposure]))
            # now overlay the images
            image_saliency_underexposure[image_saliency_underexposure > 0.1] = 255
            alpha = image_saliency_underexposure / 255.
            placement = (np.random.randint(-final_shape[0] * .50, final_shape[0] * .50, 1)[0],
                         np.random.randint(-final_shape[1] * .50, final_shape[1] * .50, 1)[0])
            final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(final_masked_blurred_image,
                                                                                 image_underexposure, placement[1],
                                                                                 placement[0], alpha)
            nMask[max(0, placement[0]) + np.argwhere(image_saliency_underexposure[y1o:y2o, x1o:x2o] == 255)[:, 0],
                  max(0, placement[1]) + np.argwhere(image_saliency_underexposure[y1o:y2o, x1o:x2o] == 255)[:, 1]] = 3
            nMask = find_overexposure_and_underexposure(np.power((final_masked_blurred_image * 1.) / 255.0,
                                                                 (1.0 / gamma)) * 255.0, nMask)

        # save image
        saveName = args.output_data_dir + "/images/"+ str(i) + "_" + args.data_extension
        final_masked_blurred_image = np.round(np.power((final_masked_blurred_image * 1.)
                                              / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
        imageio.imsave(saveName, final_masked_blurred_image)
        # save gt
        saveName = args.output_data_dir + "/gt/"+str(i) + "_" + args.data_extension
        nMask[nMask == 1] = 64
        nMask[nMask == 2] = 128
        nMask[nMask == 3] = 192
        nMask[nMask == 4] = 255
        cv2.imwrite(saveName, nMask)

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS MURI CREATE BLUR TESTING DATASET FROM FETCH')
    # directory data location
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_data_dir', type=str, default=None)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str, default=".png")
    parser.add_argument('--training_type', type=str, default=None)
    args = parser.parse_args()
    if args.training_type == '1':
        create_dataset_for_training_type_1(args)
    elif args.training_type == '2':
        create_dataset_for_training_type_2(args)
    else:
        create_dataset_for_training_type_3(args)


