import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
from config import config, log_config
from skimage import feature
from skimage import color
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageChops
import cv2
import math
from tensorflow.python.ops import array_ops
import random
import imageio
import scipy
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

def read_all_imgs(img_list, path='', n_threads=32, mode = 'RGB'):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        if mode == 'RGB':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB, path=path)
        elif mode == 'RGB2GRAY':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB2GRAY, path=path)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def get_imgs_RGB(file_name, path):
    """ Input an image path and name, return an image array """
    #https://www.codementor.io/@innat_2k14/image-data-analysis-using-numpy-opencv-part-1-kfadbafx6
    return imageio.imread(path + file_name)

def get_imgs_RGB2GRAY(file_name, path):
    """ Input an image path and name, return an image array """
    image = cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE)
    return np.asarray(image)[:,:,np.newaxis]

# crop image, flip and rotate and convert to numpy
def crop_sub_img_and_classification_fn_aug(data):

    dx = config.TRAIN.width
    dy = config.TRAIN.height
    image, mask = data
    image = np.asarray(image,dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    image_h, image_w = np.asarray(image).shape[0:2]
    if image_w != dx and dy != image_h:

        x = np.random.randint(0, image_w - dx - 1)
        y = np.random.randint(0, image_h - dy - 1)
        cropped_image = image[y: y+dy, x : x+dx, :]
        cropped_mask  = mask[y: y + dy, x: x + dx, :]
    else:
        cropped_image = image
        cropped_mask = mask
    cropped_mask = np.concatenate((cropped_mask, cropped_mask, cropped_mask), axis=2)

    rotation_list = [0,90,180,270]
    flip_list = [0,1]
    rotation= random.choice(rotation_list)
    flip= random.choice(flip_list)

    if flip == 1:
        cropped_image = cv2.flip(cropped_image,0)
        cropped_mask = cv2.flip(cropped_mask,0)
    rotation_matrix = cv2.getRotationMatrix2D((dy/2, dx/2), rotation, 1)
    cropped_image = cv2.warpAffine(cropped_image, rotation_matrix,(dy, dx))

    cropped_mask = cv2.warpAffine(cropped_mask, rotation_matrix,(dy, dx))
    return format_VGG_image(cropped_image), cropped_mask

# crop image and convert to numpy
def crop_sub_img_and_classification_fn(data):

    dx = config.TRAIN.width
    dy = config.TRAIN.height
    image, mask = data
    image = np.asarray(image,dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)

    image_h, image_w = np.asarray(image).shape[0:2]

    if image_w != dx and dy != image_h:
        x = np.random.randint(0, image_w - dx - 1)
        y = np.random.randint(0, image_h - dy - 1)
        cropped_image = image[y: y + dy, x: x + dx, :]
        cropped_mask = mask[y: y + dy, x: x + dx, :]
    else:
        cropped_image = image
        cropped_mask = mask
    cropped_mask = np.concatenate((cropped_mask, cropped_mask, cropped_mask), axis=2)
    return format_VGG_image(cropped_image), cropped_mask

def format_VGG_image(image):
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    bgr = np.zeros(image.shape)
    bgr[:, :, 0] = blue - VGG_MEAN[0]
    bgr[:, :, 1] = green - VGG_MEAN[1]
    bgr[:, :, 2] = red - VGG_MEAN[2]
    return bgr


