import numpy as np
from matplotlib import pyplot as plt
from skimage import io

gamma = 2.2

def create_brightness_and_darkness_blur(image, alpha, beta):
    new_img = image * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img

if __name__ == '__main__':
    # lets make a brightness darkness, and motion and focus blur image
    image_input = io.imread('/home/mary/code/local_success_dataset/Phos_origional/Testing/image/Phos2_uni_sc14_0.png')
    # gamma decode
    image = 255.0 * np.power((image_input * 1.) / 255.0, gamma)
    # get brightness min
    bmin = create_brightness_and_darkness_blur(image, 1.5, 0)
    # save
    bmin = np.round(np.power((np.array(bmin)[:, :, 0:3] * 1.)/ 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    io.imsave('/home/mary/code/local_success_dataset/IROS_Dataset/images_for_paper/brightness_min.png',bmin)
    # brightness max
    bmax = create_brightness_and_darkness_blur(image, 2.6, 100)
    # save
    bmax = np.round(np.power((np.array(bmax)[:, :, 0:3] * 1.) / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    io.imsave('/home/mary/code/local_success_dataset/IROS_Dataset/images_for_paper/brightness_max.png', bmax)
    # darkness min
    dmin = create_brightness_and_darkness_blur(image, 0.6, 0)
    # save
    dmin = np.round(np.power((np.array(dmin)[:, :, 0:3] * 1.) / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    io.imsave('/home/mary/code/local_success_dataset/IROS_Dataset/images_for_paper/darkness_min.png', dmin)
    # darkness max
    dmax = create_brightness_and_darkness_blur(image, 0.01, -100)
    # save
    dmax = np.round(np.power((np.array(dmax)[:, :, 0:3] * 1.) / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
    io.imsave('/home/mary/code/local_success_dataset/IROS_Dataset/images_for_paper/darkness_max.png', dmax)

