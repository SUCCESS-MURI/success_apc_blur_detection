import cv2 as cv
import imageio
import numpy as np
# hdr correction
# https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html
from misc.focus_deblurring import random_motion_blur_kernel, apply_motion_blur

if __name__ == '__main__':
    # Loading exposure images into a list
    # img_fn = ["/home/mary/frame0000.jpg", "/home/mary/frame0001.jpg",
    #           "/home/mary/frame0002.jpg", "/home/mary/frame0003.jpg",
    #           "/home/mary/frame0004.jpg", "/home/mary/frame0005.jpg",
    #           "/home/mary/frame0006.jpg", "/home/mary/frame0007.jpg",
    #           "/home/mary/frame0008.jpg"]
    img_fn = ["/home/mary/data/frame0000.jpg", "/home/mary/data/frame0003.jpg","/home/mary/data/frame0005.jpg"]

    img_list = [cv.imread(fn) for fn in img_fn]
    #exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
    image_input = 255.0 * np.power((img_list[2] * 1.) / 255.0, 2.2)
    #
    # # make a motion blur image
    kernal_size,angle = random_motion_blur_kernel()
    image_motion,motion_kernal = apply_motion_blur(image_input,kernal_size,angle)

    image_motion = (image_motion - np.min(image_motion)) * (1.0 - (0.0)) / \
                        (np.max(image_motion) - np.min(image_motion)) + 0.0

    img_list[2] = np.round(np.power((image_motion * 1.), (1.0 / 2.2)) * 255.0).astype(np.uint8)

    cv.imwrite('motion_blur_img.jpg',img_list[2])

    merge_mertens = cv.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    cv.imwrite("fusion_mertens_normal_motion.jpg", res_mertens_8bit)