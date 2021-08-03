import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
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
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_fn, path=path)
        elif mode == 'GRAY':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_GRAY_fn, path=path)
        elif mode == 'RGB2GRAY':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGBGRAY_fn, path=path)
        elif mode == 'RGB2GRAY2':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGBGRAY_2_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def get_imgs_RGB_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return imageio.imread(path + file_name)

def get_imgs_GRAY_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    # https://www.geeksforgeeks.org/python-pil-image-convert-method/
    image = Image.open(path + file_name)
    return np.array(image.convert("L"))[:,:,np.newaxis]/255

def get_imgs_GRAY_fn_cv(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    image = cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE)
    return np.expand_dims(np.asarray(image), 3)

def get_imgs_RGBGRAY_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    # https://www.geeksforgeeks.org/python-pil-image-convert-method/
    image = Image.open(path + file_name)
    return np.asarray(image)[:,:,0][:,:,np.newaxis]

def get_imgs_RGBGRAY_2_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    # https://www.geeksforgeeks.org/python-pil-image-convert-method/
    image = Image.open(path + file_name)
    return np.asarray(image)[:,:,np.newaxis]

def data_aug_train(image,mask):
    dx = config.TRAIN.width
    dy = config.TRAIN.height
    augmentation_list = [0, 1]
    augmentation = random.choice(augmentation_list)
    if augmentation == 0:
        #image, mask = crop_sub_img_and_classification_fn(image,mask)
        image_h = image.get_shape()[0]
        image_w = image.get_shape()[1]
        #TODO test this
        if image_w != dx and dy != image_h:
            x = np.random.randint(0, image_w - dx - 1)
            y = np.random.randint(0, image_h - dy - 1)
            cropped_image = tf.squeeze(tf.image.crop_and_resize(image, (y, y + dy, x, x + dx), (0), (dy, dx)))
            cropped_mask = tf.squeeze(tf.image.crop_and_resize(mask, (y, y + dy, x, x + dx), (0), (dy, dx)))
        else:
            cropped_image = image
            cropped_mask = mask
        mask = cropped_mask
    else:
        #image, mask = crop_sub_img_and_classification_fn_aug(image,mask)
        image_h = image.get_shape()[0]
        image_w = image.get_shape()[1]
        if image_w != dx and dy != image_h:
            x = np.random.randint(0, image_w - dx - 1)
            y = np.random.randint(0, image_h - dy - 1)
            cropped_image = tf.squeeze(tf.image.crop_and_resize(image, (y,y+dy,x,x+dx), (0), (dy,dx)))
            cropped_mask = tf.squeeze(tf.image.crop_and_resize(mask, (y,y+dy,x,x+dx), (0), (dy,dx)))
        else:
            cropped_image = image
            cropped_mask = mask

        rotation_list = [0, 1, 2, 3]
        flip_list = [0, 1]
        rotation = random.choice(rotation_list)
        flip = random.choice(flip_list)

        if flip == 1:
            # flip image along width x axis
            cropped_image = tf.image.flip_left_right(cropped_image)
            cropped_mask = tf.image.flip_left_right(cropped_mask)
        if rotation != 0:
            # https://www.tensorflow.org/api_docs/python/tf/image/rot90
            cropped_image = tf.image.rot90(cropped_image, k=rotation)
            cropped_mask = tf.image.rot90(cropped_mask, k=rotation)
    mask = tf.cast(cropped_mask,dtype=tf.float32)
    red, green, blue = tf.split(cropped_image,3,2)
    bgr = tf.concat([blue - VGG_MEAN[0],green - VGG_MEAN[1],red - VGG_MEAN[2],], axis=2)
    return bgr, mask

def data_aug_valid(image, mask):
    dx = config.TRAIN.width
    dy = config.TRAIN.height
    # image, mask = crop_sub_img_and_classification_fn(image,mask)
    image_h = image.get_shape()[0]
    image_w = image.get_shape()[1]
    # TODO Test this
    if image_w != dx and dy != image_h:
        x = np.random.randint(0, image_w - dx - 1)
        y = np.random.randint(0, image_h - dy - 1)
        cropped_image = tf.squeeze(tf.image.crop_and_resize(image, (y, y + dy, x, x + dx), (0), (dy, dx)))
        cropped_mask = tf.squeeze(tf.image.crop_and_resize(mask, (y, y + dy, x, x + dx), (0), (dy, dx)))
    else:
        cropped_image = image
        cropped_mask = mask
    mask = tf.cast(cropped_mask,dtype=tf.float32)
    red, green, blue = tf.split(cropped_image, 3, 2)
    bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2], ], axis=2)
    return bgr, mask

def crop_sub_img_and_classification_fn_aug(data):

    dx = config.TRAIN.width
    dy = config.TRAIN.height
    # image = image.eval(sess)
    # mask = mask.eval(sess)
    image, mask = data
    #print "image shape", image.shape
    #print "mask shape", mask.shape
    image = np.asarray(image,dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)

    image_h, image_w = np.asarray(image).shape[0:2]
    if image_w != dx and dy != image_h:

        x = np.random.randint(0, image_w - dx - 1)
        y = np.random.randint(0, image_h - dy - 1)
        cropped_image = image[y: y+dy, x : x+dx, :]
        #print "hi2"
        cropped_mask  = mask[y: y + dy, x: x + dx, :]
    else:
        cropped_image = image
        cropped_mask = mask
    cropped_mask = np.concatenate((cropped_mask, cropped_mask, cropped_mask), axis=2)

    rotation_list = [0,90,180,270]
    flip_list = [0,1]
    rotation= random.choice(rotation_list)
    flip= random.choice(flip_list)

    if(flip ==1):
        cropped_image = cv2.flip(cropped_image,0)
        cropped_mask = cv2.flip(cropped_mask,0)
    rotation_matrix = cv2.getRotationMatrix2D((dy/2, dx/2), rotation, 1)
    cropped_image = cv2.warpAffine(cropped_image, rotation_matrix,(dy, dx))

    cropped_mask = cv2.warpAffine(cropped_mask, rotation_matrix,(dy, dx))
    #scipy.misc.imsave('samples/input_image.png', cropped_image)
    #scipy.misc.imsave('samples/input_mask.png', cropped_mask[:,:,0])

    #print edge.shape

    #print "hi3"
    # cropped_mask = np.expand_dims(cropped_mask, axis=3)


    #score = (np.sum(cropped_mask) / (dx * dy))

    #print "image shape", cropped_image.shape
   # print "mask shape", cropped_mask.shape

    #cropped_image = np.zeros((192,192,3))
    #cropped_mask = np.zeros((192,192,1))


    return format_VGG_image(cropped_image), cropped_mask

def crop_sub_img_and_classification_fn(data):

    dx = config.TRAIN.width
    dy = config.TRAIN.height
    # image = image.eval(sess)
    # mask = mask.eval(sess)
    image, mask = data
    #print "image shape", image.shape
    #print "mask shape", mask.shape
    image = np.asarray(image,dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)

    image_h, image_w = np.asarray(image).shape[0:2]

    if image_w != dx and dy != image_h:

        x = np.random.randint(0, image_w - dx - 1)
        y = np.random.randint(0, image_h - dy - 1)
        cropped_image = image[y: y + dy, x: x + dx, :]
        # print "hi2"
        cropped_mask = mask[y: y + dy, x: x + dx, :]
    else:
        cropped_image = image
        cropped_mask = mask
    cropped_mask = np.concatenate((cropped_mask, cropped_mask, cropped_mask), axis=2)

    #print edge.shape

    #print "hi3"
    # cropped_mask = np.expand_dims(cropped_mask, axis=3)


    #score = (np.sum(cropped_mask) / (dx * dy))

    #print "image shape", cropped_image.shape
   # print "mask shape", cropped_mask.shape

    #cropped_image = np.zeros((192,192,3))
    #cropped_mask = np.zeros((192,192,1))


    return format_VGG_image(cropped_image), cropped_mask

def format_VGG_image(image):
    #rgb_scaled = image * 255.0
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    bgr = np.zeros(image.shape)
    bgr[:, :, 0] = blue - VGG_MEAN[0]
    bgr[:, :, 1] = green - VGG_MEAN[1]
    bgr[:, :, 2] = red - VGG_MEAN[2]
    #bgr = np.round(bgr).astype(np.float32)
    return bgr

# def unformat_VGG_Image(image):
#     blue = image[:, :, 0]
#     green = image[:, :, 1]
#     red = image[:, :, 2]
#     bgr = np.zeros(image.shape)
#     bgr[:, :, 2] = blue + VGG_MEAN[0]
#     bgr[:, :, 1] = green + VGG_MEAN[1]
#     bgr[:, :, 0] = red + VGG_MEAN[2]
#     #bgr = np.round(bgr).astype(np.float32)
#     return bgr

