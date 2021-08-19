import tensorlayer as tl
from config import config
from PIL import Image
import cv2
import random
import imageio
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

def read_all_imgs(img_list, path='', n_threads=32, mode = 'RGB'):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        if mode == 'RGB':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_cv2, path=path)
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

# def get_imgs_RGB_fn(file_name, path):
#     """ Input an image path and name, return an image array """
#     # return scipy.misc.imread(path + file_name).astype(np.float)
#     return imageio.imread(path + file_name)

def get_imgs_RGB_cv2(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return cv2.cvtColor(cv2.imread(path+file_name),cv2.COLOR_BGR2RGB) #cv2.COLORBGRTORGB

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
    image = cv2.imread(path+file_name)
    return np.asarray(image)[:,:,0][:,:,np.newaxis]

def crop_sub_img_and_classification_fn_aug(data):

    dx = config.TRAIN.width
    dy = config.TRAIN.height
    # image = image.eval(sess)
    # mask = mask.eval(sess)
    image, mask = data

    image_h, image_w = image.shape[0:2]
    if image_w != dx and dy != image_h:
        # x = np.random.randint(0, image_w - dx - 1)
        # y = np.random.randint(0, image_h - dy - 1)
        # cropped_image = image[y: y+dy, x : x+dx, :]
        # #print "hi2"
        # cropped_mask  = mask[y: y + dy, x: x + dx, :]
        cropped_image = cv2.resize(image, [dy, dx], interpolation=cv2.INTER_NEAREST)
        cropped_mask = cv2.resize(mask, [dy, dx], interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]
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
    return format_VGG_image(cropped_image), cropped_mask

def crop_sub_img_and_classification_fn(data):

    dx = config.TRAIN.width
    dy = config.TRAIN.height
    # image = image.eval(sess)
    # mask = mask.eval(sess)
    image, mask = data
    #print "image shape", image.shape
    #print "mask shape", mask.shape

    image_h, image_w = image.shape[0:2]

    if image_w != dx and dy != image_h:
        # x = np.random.randint(0, image_w - dx - 1)
        # y = np.random.randint(0, image_h - dy - 1)
        # cropped_image = image[y: y+dy, x : x+dx, :]
        # #print "hi2"
        # cropped_mask  = mask[y: y + dy, x: x + dx, :]
        cropped_image = cv2.resize(image, [dy, dx], interpolation=cv2.INTER_NEAREST)
        cropped_mask = cv2.resize(mask, [dy, dx], interpolation=cv2.INTER_NEAREST)[:,:,np.newaxis]
    else:
        cropped_image = image
        cropped_mask = mask
    cropped_mask = np.concatenate((cropped_mask, cropped_mask, cropped_mask), axis=2)
    return format_VGG_image(cropped_image), cropped_mask

def format_VGG_image(image):
    rgb_scaled = image * 1.0
    red = rgb_scaled[:, :, 0]
    green = rgb_scaled[:, :, 1]
    blue = rgb_scaled[:, :, 2]
    bgr = np.zeros(image.shape)
    bgr[:, :, 0] = blue - VGG_MEAN[0]
    bgr[:, :, 1] = green - VGG_MEAN[1]
    bgr[:, :, 2] = red - VGG_MEAN[2]
    return bgr


