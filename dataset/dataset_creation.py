# Author mhatfalv
# Create the blur dataset
import argparse
import copy
import glob
import random

import cv2
import numpy as np
import saliency_detection
from PIL import Image as im, ImageEnhance
import tensorlayer as tl
import matplotlib.pyplot as plt

# create motion blur
# https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
from utils import read_all_imgs

gamma = 2.5

def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    return cv2.filter2D(image, -1, k)

# create out of focus blur for 3 channel images
def create_out_of_focus_blur(image,kernelsize):
    image_blurred = cv2.blur(image,(kernelsize,kernelsize))
    return image_blurred

def create_brightness_and_darkness_blur(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

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

def create_muri_dataset_for_training(args):
    # angles range for dataset
    angles = np.arange(start=0, stop=190, step=30)
    kernal = np.arange(start=9, stop=26, step=4)
    # alpha
    alpha_darkness = np.arange(start=0.1, stop=0.6, step=0.1)
    alpha_brightness = np.arange(start=1.9, stop=2.5, step=0.1)
    # beta
    #beta = np.arange(start=0,stop=1,step=1) # 110 step 10
    final_shape = (480,640)
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
        new_image = np.zeros((final_shape[0],final_shape[1],3))
        new_mask = np.zeros((final_shape[0],final_shape[1],1))
        padd_h = int((final_shape[0] +1 - imagesOrigonal[i].shape[0])/2)
        padd_w = int((final_shape[1] +1- imagesOrigonal[i].shape[1])/2)
        new_image[padd_h:imagesOrigonal[i].shape[0]+padd_h,padd_w:imagesOrigonal[i].shape[1]+padd_w] = imagesOrigonal[i]
        new_mask[padd_h:imagesOrigonal[i].shape[0]+padd_h,padd_w:imagesOrigonal[i].shape[1]+padd_w] = SaliencyImages[i]
        # gamma correction
        imagesOrigonal[i] = 255.0*np.power((new_image*1.)/255,gamma)
        (T, saliency) = cv2.threshold(new_mask, 1, 1, cv2.THRESH_BINARY)
        SaliencyImages[i] = saliency
        maskSaliency.append(saliency[:, :, np.newaxis] * imagesOrigonal[i])
        images_list[i] = images_list[i].split('.')[0]

    # now we will go through all of the images and make the dataset
    indexs = np.arange(start=0,stop=len(imagesOrigonal),step=1)
    np.random.shuffle(indexs)
    idxs = np.arange(start=0,stop=4,step=1)
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
                                        np.copy(maskSaliency[overlayImageIdx]),s,a)
                                    # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
                                    # cv2.waitKey(0)
                                    blur_mask = apply_motion_blur(np.copy(SaliencyImages[overlayImageIdx])*255,s,a)
                                    motion_blur_mask = np.copy(blur_mask)
                                    motion_blur_mask[np.round(blur_mask,4) > 0] = 255
                                    alpha = blur_mask / 255.
                                    # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
                                    # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
                                    # motion_blurred_imgPIL,alpha_PIL = convert_whiten_image(motion_blurred_overlay_img,
                                    #                                         im.fromarray(alpha).convert('RGBA'))
                                    placement = (np.random.randint(-final_shape[0]*.75,final_shape[0]*.75,1)[0],
                                         np.random.randint(-final_shape[1]*.75,final_shape[1]*.75,1)[0])
                                    final_masked_blurred_image,y1o,y2o,x1o,x2o = overlay_image_alpha(
                                        final_masked_blurred_image,motion_blurred_overlay_img,placement[1],placement[0],
                                        alpha)
                                    # # https://note.nkmk.me/en/python-pillow-paste/
                                    # final_masked_blurred_image.paste(motion_blurred_imgPIL, placement, mask=alpha_PIL)
                                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    nMask[max(0,placement[0])+np.argwhere(motion_blur_mask[y1o:y2o,x1o:x2o]==255)[:,0],
                                        max(0,placement[1])+np.argwhere(motion_blur_mask[y1o:y2o,x1o:x2o]==255)[:,1]]=64
                                    # cv2.imshow('BrightnessImage', nMask)
                                    # cv2.waitKey(0)
                                    # focus blur
                                elif j == 1:
                                    # now do out of focus blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    focus_blurred_overlay_img = create_out_of_focus_blur(
                                        np.copy(maskSaliency[overlayImageIdx]),k)
                                    blur_mask = create_out_of_focus_blur(np.copy(SaliencyImages[overlayImageIdx])*255,k)
                                    focus_blur_mask = np.copy(blur_mask)
                                    focus_blur_mask[np.round(blur_mask, 4) > 0] = 255
                                    alpha = blur_mask / 255.
                                    # focus_blurred_imgPIL, alpha_PIL = convert_whiten_image(focus_blurred_overlay_img,
                                    #                                        im.fromarray(alpha).convert('RGBA'))
                                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(
                                        final_masked_blurred_image,focus_blurred_overlay_img,placement[1],placement[0],
                                        alpha)
                                    # final_masked_blurred_image.paste(focus_blurred_imgPIL, placement,
                                    #                                  mask=alpha_PIL)
                                    # cv2.imshow('FocusImage',final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add focus identification to mask
                                    #idx = np.where(np.array(alpha_PIL)[:, :, 3] == 255)
                                    nMask[max(0,placement[0])+np.argwhere(focus_blur_mask[y1o:y2o,x1o:x2o]==255)[:,0],
                                        max(0,placement[1])+np.argwhere(focus_blur_mask[y1o:y2o,x1o:x2o]==255)[:,1]]=128
                                # darkness blur
                                elif j == 2:
                                    #create darkness blur
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
                                    #darkness = True
                                    # dark_blurred_imgPIL,alpha_PIL = convert_whiten_image(dark_blurred_overlay_img,
                                    #                                             im.fromarray(alpha).convert('RGBA'))
                                    # # cv2.imshow('BrightnessImage2', np.array(dark_blurred_imgPIL)[:,:,0:3])
                                    # # cv2.waitKey(0)
                                    placement = (np.random.randint(-final_shape[0] * .75, final_shape[0] * .75, 1)[0],
                                                 np.random.randint(-final_shape[1] * .75, final_shape[1] * .75, 1)[0])
                                    final_masked_blurred_image, y1o, y2o, x1o, x2o = overlay_image_alpha(
                                        final_masked_blurred_image, dark_blurred_overlay_img, placement[1],placement[0],
                                        alpha)
                                    # final_masked_blurred_image.paste(dark_blurred_imgPIL, placement,
                                    #                                 mask=alpha_PIL)

                                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add brightness and/or darkness blur to mask image
                                    # indicator for dark blur
                                    nMask[max(0,placement[0])+np.argwhere(dark_blur_mask[y1o:y2o,x1o:x2o]==255)[:,0],
                                          max(0,placement[1])+np.argwhere(dark_blur_mask[y1o:y2o,x1o:x2o]==255)[:,1]]=192
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
                                        placement[0],alpha)
                                    #final_masked_blurred_image.paste(bright_blurred_imgPIL, placement,mask=alpha_PIL)
                                    # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add brightness and/or darkness blur to mask image
                                    #idx = np.where(np.array(alpha_PIL)[:, :, 3] == 255)
                                    # indicator for brightness
                                    nMask[max(0,placement[0])+np.argwhere(bright_blur_mask[y1o:y2o,x1o:x2o]==255)[:,0],
                                          max(0,placement[1])+np.argwhere(bright_blur_mask[y1o:y2o,x1o:x2o]==255)[:,1]]=255
                            # save final image
                            saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur" \
                                    + "_a_" + str(a)  + "_s_" + str(s) + "_focus_blur_k_" + str(k) \
                                    + "_darkness_blur_al_" + str(al_d) + "_brightness_blur_al_" + str(al_b) + \
                                   args.data_extension
                            final_masked_blurred_image = np.power((np.array(final_masked_blurred_image)[:, :, 0:3]*1.)
                                                                  /255,(1/gamma))*255.0
                            cv2.imwrite(saveName, final_masked_blurred_image)
                            # create and save ground truth mask
                            saveNameGt = args.output_data_dir + "/gt/" + baseImageName + "_motion_blur" \
                                    + "_a_" + str(a)  + "_s_" + str(s) + "_focus_blur_k_" + str(k) \
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
                motion_blur_mask[np.round(blur_mask,4) > 0] = 255
                alpha = blur_mask / 255.0
                final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]),motion_blurred_overlay_img,
                                                            alpha[:,:,np.newaxis])
                # now save the image
                saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                cv2.imwrite(saveName,np.power((final_masked_blurred_image*1.)/255,(1/gamma))*255.0)
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
            focus_blur_mask[np.round(blur_mask,4) > 0] = 255
            alpha = blur_mask / 255.
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), focus_blur_overlay_img,
                                                        alpha[:,:,np.newaxis])
            saveName = args.output_data_dir + "/images/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            cv2.imwrite(saveName,np.power((final_masked_blurred_image*1.)/255,(1/gamma))*255.0)
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
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]), dark_blur, alpha[:,:,np.newaxis])
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image*1.)/255,(1/gamma))*255.0)
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
            final_masked_blurred_image = alpha_blending(np.copy(imagesOrigonal[idx]),bright_blur,alpha[:,:,np.newaxis])
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, np.power((final_masked_blurred_image*1.)/255,(1/gamma))*255.0)
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
            baseImage = np.power((imagesOrigonal[i]*1.)/255,(1/gamma))*255.0
            baseImageName = images_list[i]
            saveName = args.output_data_dir + "/images/" + baseImageName + "_no_blur_" + str(
                n + 1) + args.data_extension
            cv2.imwrite(saveName, baseImage)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_no_blur_" + str(
                n + 1) + args.data_extension
            nMask = np.zeros(final_shape)
            cv2.imwrite(saveName, nMask)

def create_muri_dataset_for_testing_and_validation(args):
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

# def create_muri_dataset_for_sensitivity(args):
#     # angles range for dataset
#     angles = np.arange(start=0,stop=190,step=10)
#     kernalSize = np.array([3,5,7,9,13,15,17,19,21,23])
#     # kernal
#     kernal = np.array([3,5,7,9,13,15,17,19,21,23])
#     # alpha
#     alpha_darkness = np.arange(start=0.05, stop=0.7, step=0.05)
#     alpha_brightness = np.arange(start=1.6, stop=2.0, step=0.05)
#     #alpha_brightness = np.arange(start=1.8, stop=2.3, step=0.05)
#     #alpha_brightness = np.arange(start=200, stop=255, step=10)
#     # beta
#     #beta = np.arange(start=0,stop=1,step=1) # 110 step 10
#     tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
#     tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
#     # list of all original Images
#     imagesOrigonal = []
#     imageNames = []
#     imageOrigMask = []
#     imagesAll = []
#     imagesAllNames = []
#     saliency = saliency_detection.Saliency_NN()
#     for imageFileName in glob.glob(args.data_dir + '/*'+ args.data_extension):
#         image = cv2.imread(imageFileName)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         imageNameSplit = imageFileName.split('/')
#         imageNameSplit2 = imageNameSplit[-1].split(".")
#         # store all of the images and the mask of the saliency image with the image name
#         imagesAll.append(image)
#         # cv2.imshow('BrightnessImage', image)
#         # cv2.waitKey(0)
#         # https://stackoverflow.com/questions/67258207/how-can-i-join-strings-within-a-list-python
#         imagesAllNames.append(''.join(imageNameSplit2[0:2]))
#         if 'noblock' in imageFileName:
#             continue
#         imagesOrigonal.append(image)
#         imageNames.append(''.join(imageNameSplit2[0:2]))
#         saliency_image = saliency.compute_saliency_NN(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
#         imageOrigMask.append(saliency_image)
#         # cv2.imshow('saliency_image', image*saliency_image[:,:,np.newaxis])
#         # cv2.waitKey(0)
#     # now we will go through all of the images and make the dataset
#     indexs = np.arange(start=0,stop=len(imagesOrigonal),step=1)
#     np.random.shuffle(indexs)
#     for i in range(len(imagesOrigonal)):
#         #tempIdxs = indexs[indexs != i]
#         idx = indexs[i]
#         baseImageName = imageNames[idx]
#         for s in kernalSize:
#             for a in angles:
#                 # motion blur
#                 overlayImageIdx = idx  # np.random.choice(tempIdxs, 1, replace=False)[0]
#                 overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
#                 motion_blurred_overlay_img = apply_motion_blur(overlayImage, s, a)
#                 mask = imageOrigMask[overlayImageIdx]
#                 # saliency.compute_saliency_NN(copy.deepcopy(motion_blurred_overlay_img))
#                 motion_blurred_img = motion_blurred_overlay_img * mask[:, :, np.newaxis]
#                 maskedImageRemoved = imagesOrigonal[overlayImageIdx] * np.logical_not(mask[:, :, np.newaxis])
#                 final_masked_blurred_image = maskedImageRemoved + motion_blurred_img
#                 # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
#                 # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
#                 # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
#                 # motion_blurred_imgPIL = convert_whiten_and_crop_image(motion_blurred_img, False)
#                 # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
#                 #             np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
#                 # maskedImageRemovedPIL.paste(motion_blurred_imgPIL, placement, mask=motion_blurred_imgPIL)
#                 # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]  # + motion_blurred_img
#                 # now save the image
#                 saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
#                            str(s) + args.data_extension
#                 cv2.imwrite(saveName, final_masked_blurred_image)
#                 saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_motion_blur_a_" + str(a) + "_s_" + \
#                            str(s) + args.data_extension
#                 # cv2.imshow('BrightnessImage', final_masked_blurred_image)
#                 # cv2.waitKey(0)
#                 nMask = np.zeros(mask.shape)
#                 nMask[mask == 1] = 64
#                 # idx = np.where(np.array(motion_blurred_imgPIL)[:, :, 3] == 255)
#                 # idx_x = np.minimum(placement[1] + idx[0], np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
#                 # idx_y = np.minimum(placement[0] + idx[1], np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
#                 # nMask[idx_x, idx_y] = 64
#                 cv2.imwrite(saveName, nMask)
#                 # cv2.imshow('BrightnessImage', nMask)
#                 # cv2.waitKey(0)
#         ### out of focus blur
#         for k in kernal:
#             overlayImageIdx = idx  # np.random.choice(tempIdxs, 1, replace=False)[0]
#             overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
#             focus_blur = create_out_of_focus_blur(overlayImage, k)
#             mask = imageOrigMask[overlayImageIdx]
#             focus_blurred_img = focus_blur * mask[:, :, np.newaxis]
#             maskedImageRemoved = imagesOrigonal[overlayImageIdx] * np.logical_not(mask[:, :, np.newaxis])
#             final_masked_blurred_image = maskedImageRemoved + focus_blurred_img
#             # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
#             # focus_blurred_imgPIL = convert_whiten_and_crop_image(focus_blurred_img, False)
#             # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
#             #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
#             # maskedImageRemovedPIL.paste(focus_blurred_imgPIL, placement, mask=focus_blurred_imgPIL)
#             # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
#             # cv2.imshow('BrightnessImage',final_masked_blurred_image)
#             # cv2.waitKey(0)
#             saveName = args.output_data_dir + "/images/" + baseImageName + "_focus_blur_k_" + str(k) + \
#                        args.data_extension
#             cv2.imwrite(saveName, final_masked_blurred_image)
#             # get ground truth mask
#             saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_focus_blur_k_" + str(k) + \
#                        args.data_extension
#             # add focus identification to mask
#             nMask = np.zeros(mask.shape)
#             nMask[mask == 1] = 128
#             # idx = np.where(np.array(focus_blurred_imgPIL)[:, :, 3] == 255)
#             # idx_x = np.minimum(placement[1] + idx[0],
#             #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
#             # idx_y = np.minimum(placement[0] + idx[1],
#             #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
#             # nMask[idx_x, idx_y] = 128
#             cv2.imwrite(saveName, nMask)
#         # next we go to darkness blur
#         for a in alpha_darkness:
#             # create brightness/darkness blur
#             overlayImageIdx = idx
#             overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
#             mask = imageOrigMask[overlayImageIdx]
#             dark_blur = create_darkness_blur(overlayImage, a, 0)
#             dark_blur_masked = dark_blur * mask[:, :, np.newaxis]
#             maskedImageRemoved = imagesOrigonal[overlayImageIdx] * np.logical_not(mask[:, :, np.newaxis])
#             final_masked_blurred_image = maskedImageRemoved + dark_blur_masked
#             # dark_blur_masked[imageMask == 0] = 255
#             # darkness = True
#             # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
#             # dark_blurred_imgPIL = convert_whiten_and_crop_image(dark_blur_masked, darkness)
#             # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
#             #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
#             # maskedImageRemovedPIL.paste(dark_blurred_imgPIL, placement, mask=dark_blurred_imgPIL)
#             # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
#             # cv2.imshow('BrightnessImage', final_masked_blurred_image)
#             # cv2.waitKey(0)
#             # save the image
#             saveName = args.output_data_dir + "/images/" + baseImageName + "_darkness_blur_al_" + str(a) + \
#                        args.data_extension
#             cv2.imwrite(saveName, final_masked_blurred_image)
#             # get ground truth mask
#             saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_darkness_blur_al_" + str(a) + \
#                        args.data_extension
#             nMask = np.zeros(mask.shape)
#             nMask[mask == 1] = 192
#             # add brightness and/or darkness blur to mask image
#             # idx = np.where(np.array(dark_blurred_imgPIL)[:, :, 3] == 255)
#             # idx_x = np.minimum(placement[1] + idx[0],
#             #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
#             # idx_y = np.minimum(placement[0] + idx[1],
#             #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
#             # indicator for dark blur
#             # nMask[idx_x, idx_y] = 192
#             # else:  # indicator for brightness
#             #     nMask[idx_x, idx_y] = 255
#             cv2.imwrite(saveName, nMask)
#         # next we go to brightness blur
#         for a in alpha_brightness:
#             # create brightness blur
#             overlayImageIdx = idx  # np.random.choice(tempIdxs, 1, replace=False)[0]
#             overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
#             mask = imageOrigMask[overlayImageIdx]
#             bright_blur = create_darkness_blur(overlayImage, a, 0)
#             bright_blur_masked = bright_blur * mask[:, :, np.newaxis]
#             maskedImageRemoved = imagesOrigonal[overlayImageIdx] * np.logical_not(mask[:, :, np.newaxis])
#             final_masked_blurred_image = maskedImageRemoved + bright_blur_masked
#             # darkness = False
#             # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
#             # bright_blurred_imgPIL = convert_whiten_and_crop_image(bright_blur_masked, darkness)
#             # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
#             #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
#             # maskedImageRemovedPIL.paste(bright_blurred_imgPIL, placement, mask=bright_blurred_imgPIL)
#             # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
#             # cv2.imshow('BrightnessImage', final_masked_blurred_image)
#             # cv2.waitKey(0)
#             # save the image
#             saveName = args.output_data_dir + "/images/" + baseImageName + "_brightness_blur_al_" + str(a) + \
#                        args.data_extension
#             cv2.imwrite(saveName, final_masked_blurred_image)
#             # get ground truth mask
#             saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_brightness_blur_al_" + str(a) + \
#                        args.data_extension
#             nMask = np.zeros(mask.shape)
#             nMask[mask == 1] = 255
#             # add brightness and/or darkness blur to mask image
#             # idx = np.where(np.array(bright_blurred_imgPIL)[:, :, 3] == 255)
#             # idx_x = np.minimum(placement[1] + idx[0],
#             #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
#             # idx_y = np.minimum(placement[0] + idx[1],
#             #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
#             # indicator for brightness
#             # nMask[idx_x, idx_y] = 255
#             cv2.imwrite(saveName, nMask)
#     # save control / no blur images
#     for i in range(len(imagesAll)):
#         baseImage = imagesAll[i]
#         baseImageName = imagesAllNames[i]
#         saveName = args.output_data_dir + "/images/" + baseImageName + "_no_blur" + args.data_extension
#         cv2.imwrite(saveName, baseImage)
#         # get ground truth mask
#         saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_no_blur" + args.data_extension
#         nMask = np.zeros(mask.shape)
#         cv2.imwrite(saveName, nMask)

if __name__ == "__main__":
 # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS MURI CREATE BLUR DATASET')
    # directory data location
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_data_dir', type=str)
    parser.add_argument('--salinecy_data_dir', type=str)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str,default=".png")
    parser.add_argument('--is_testing', default=False, action='store_true')
    args = parser.parse_args()
    if args.is_testing:
        create_muri_dataset_for_testing_and_validation(args)
       # create_muri_dataset_for_sensitivity(args)
    else:
        create_muri_dataset_for_training(args)


