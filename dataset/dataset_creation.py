# Author mhatfalv
# Create the blur dataset
import argparse
import copy
import glob
import random

import cv2
import numpy as np
import saliency_detection
from PIL import Image as im
import tensorlayer as tl

# create motion blur
def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
    k = k * ( 1.0 / np.sum(k) )
    return cv2.filter2D(image, -1, k)

# create oout of focus blur for 3 channel images
def create_out_of_focus_blur(image,kernelsize):
    image_blurred = cv2.blur(image,(kernelsize,kernelsize))
    return image_blurred

# make image brighter or darker based on image changes
def create_brightness_blur(image,alpha,beta):
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # enhancer = ImageEnhance.Brightness(im.fromarray(np.uint8(image)))
    # new_image = np.array(enhancer.enhance(beta))
    return new_image

def convert_whiten_and_crop_image(image,darkness):
    imgPIL = im.fromarray(np.uint8(image)).convert('RGBA')
    datas = imgPIL.getdata()
    # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
    # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255 and darkness:
            newData.append((255, 255, 255, 0))
        elif item[0] == 0 and item[1] == 0 and item[2] == 0 and ~darkness:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)

    imgPIL.putdata(newData)
    # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
    imageBox = imgPIL.getbbox()
    imgPIL = imgPIL.crop(imageBox)
    return imgPIL

def create_muri_dataset_for_training(args):
    # angles range for dataset
    angles = np.arange(start=0,stop=190,step=30)
    kernalSize = np.array([3, 5, 7, 9])
    # kernal
    kernal = np.array([3,5,7,9])
    # alpha
    alpha_darkness = np.arange(start=0.1, stop=0.6, step=0.1)
    alpha_brightness = np.arange(start=1.5, stop=2.0, step=0.1)
    # beta
    #beta = np.arange(start=0,stop=1,step=1) # 110 step 10
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    imagesOrigonal = []
    imageNames = []
    imageOrigMask = []
    imagesAll = []
    imagesAllNames = []
    saliency = saliency_detection.Saliency_NN()
    for imageFileName in glob.glob(args.data_dir + '/*'+ args.data_extension):
        image = cv2.imread(imageFileName)
        imageNameSplit = imageFileName.split('/')
        imageNameSplit2 = imageNameSplit[-1].split(".")
        # store all of the images and the mask of the saliency image with the image name
        imagesAll.append(image)
        # cv2.imshow('BrightnessImage', image)
        # cv2.waitKey(0)
        # https://stackoverflow.com/questions/67258207/how-can-i-join-strings-within-a-list-python
        imagesAllNames.append(''.join(imageNameSplit2[0:2]))
        if 'noblock' in imageFileName:
            continue
        imagesOrigonal.append(image)
        imageNames.append(''.join(imageNameSplit2[0:2]))
        saliency_image = saliency.compute_saliency_NN(copy.deepcopy(image))
        imageOrigMask.append(saliency_image)
        # cv2.imshow('saliency_image', image*saliency_image[:,:,np.newaxis])
        # cv2.waitKey(0)
    # now we will go through all of the images and make the dataset
    indexs = np.arange(start=0,stop=len(imagesOrigonal),step=1)
    np.random.shuffle(indexs)
    idxs = np.arange(start=0,stop=4,step=1)
    # this already makes a huge dataset
    for i in range(1):
        tempIdxs = indexs[indexs != i]
        baseImage = imagesOrigonal[i]
        baseImageName = imageNames[i]
        for s in kernalSize:
            for a in angles:
                ### out of focus blur
                for k in kernal:
                    for al_d in alpha_darkness:
                        for al_b in alpha_brightness:
                            final_masked_blurred_image = baseImage
                            nMask = np.zeros(baseImage.shape[0:2])
                            np.random.shuffle(idxs)
                            for j in idxs:
                                # motion blur
                                if j == 0:
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                                    motion_blurred_overlay_img = apply_motion_blur(overlayImage,s,a)
                                    # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
                                    # cv2.waitKey(0)
                                    mask = imageOrigMask[overlayImageIdx]#saliency.compute_saliency_NN(copy.deepcopy(motion_blurred_overlay_img))
                                    motion_blurred_img = motion_blurred_overlay_img * mask[:, :, np.newaxis]
                                    #motion_blurred_img[motion_blurred_img == 0] = 255
                                    # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
                                    # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
                                    maskedImageRemovedPIL = im.fromarray(np.uint8(final_masked_blurred_image)).convert('RGBA')
                                    motion_blurred_imgPIL = convert_whiten_and_crop_image(motion_blurred_img,False)
                                    placement = (np.random.randint(0,baseImage.shape[0]*.75,1)[0],
                                             np.random.randint(0,baseImage.shape[1]*.75,1)[0])
                                    maskedImageRemovedPIL.paste(motion_blurred_imgPIL, placement,mask=motion_blurred_imgPIL)
                                    final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:,:,0:3]
                                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    idx = np.where(np.array(motion_blurred_imgPIL)[:,:,3] == 255)
                                    idx_x = np.minimum(placement[1] + idx[0],
                                                   np.ones(idx[0].shape[0]) * (nMask.shape[0]-1)).astype(int)
                                    idx_y = np.minimum(placement[0] + idx[1],
                                                    np.ones(idx[1].shape[0]) * (nMask.shape[1]-1)).astype(int)
                                    nMask[idx_x, idx_y] = 64
                                    # cv2.imshow('BrightnessImage', nMask)
                                    # cv2.waitKey(0)
                                # focus blur
                                elif j == 1:
                                    # now do out of focus blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                                    focus_blur = create_out_of_focus_blur(overlayImage, k)
                                    mask_focus = imageOrigMask[overlayImageIdx]#saliency.compute_saliency_NN(copy.deepcopy(focus_blur))
                                    focus_blurred_img = focus_blur * mask_focus[:, :, np.newaxis]
                                    #focus_blurred_img[focus_blurred_img == 0] = 255
                                    maskedImageRemovedPIL = im.fromarray(np.uint8(final_masked_blurred_image)).convert('RGBA')
                                    focus_blurred_imgPIL = convert_whiten_and_crop_image(focus_blurred_img,False)
                                    placement = (np.random.randint(0, baseImage.shape[0]*.75, 1)[0],
                                                np.random.randint(0, baseImage.shape[1]*.75, 1)[0])
                                    maskedImageRemovedPIL.paste(focus_blurred_imgPIL, placement,mask=focus_blurred_imgPIL)
                                    final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
                                    # cv2.imshow('FocusImage',final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add focus identification to mask
                                    idx = np.where(np.array(focus_blurred_imgPIL)[:, :, 3] == 255)
                                    idx_x = np.minimum(placement[1] + idx[0],
                                                    np.ones(idx[0].shape[0])*(nMask.shape[0]-1)).astype(int)
                                    idx_y = np.minimum(placement[0] + idx[1],
                                                    np.ones(idx[1].shape[0])*(nMask.shape[1]-1)).astype(int)
                                    nMask[idx_x, idx_y] = 128
                                # darkness blur
                                elif j == 2:
                                    # create darkness blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                                    imageMask = imageOrigMask[overlayImageIdx]
                                    dark_blur = create_brightness_blur(overlayImage, al_d,0)
                                    # cv2.imshow('BrightnessImage', dark_blur)
                                    # cv2.waitKey(0)
                                    dark_blur_masked = dark_blur * imageMask[:, :, np.newaxis]
                                    dark_blur_masked[imageMask == 0] = 255
                                    # cv2.imshow('BrightnessImage_mask', dark_blur_masked)
                                    # cv2.waitKey(0)
                                    darkness = True
                                    maskedImageRemovedPIL = im.fromarray(np.uint8(final_masked_blurred_image)).\
                                            convert('RGBA')
                                    dark_blurred_imgPIL = convert_whiten_and_crop_image(dark_blur_masked,darkness)
                                    # cv2.imshow('BrightnessImage2', np.array(dark_blurred_imgPIL)[:,:,0:3])
                                    # cv2.waitKey(0)
                                    placement = (np.random.randint(0, baseImage.shape[0]*.75, 1)[0],
                                                np.random.randint(0, baseImage.shape[1]*.75, 1)[0])
                                    maskedImageRemovedPIL.paste(dark_blurred_imgPIL, placement,
                                                                    mask=dark_blurred_imgPIL)
                                    final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
                                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add brightness and/or darkness blur to mask image
                                    idx = np.where(np.array(dark_blurred_imgPIL)[:, :, 3] == 255)
                                    idx_x = np.minimum(placement[1] + idx[0],
                                                    np.ones(idx[0].shape[0]) * (nMask.shape[0]-1)).astype(int)
                                    idx_y = np.minimum(placement[0] + idx[1],
                                                    np.ones(idx[1].shape[0]) * (nMask.shape[1]-1)).astype(int)
                                    # indicator for dark blur
                                    nMask[idx_x, idx_y] = 192
                                # brightness blur
                                elif j == 3:
                                    # create brightness blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                                    imageMask = imageOrigMask[overlayImageIdx]
                                    brightness_blur = create_brightness_blur(overlayImage, al_b,0)
                                    brightness_blur_masked = brightness_blur * imageMask[:, :, np.newaxis]
                                    darkness = False
                                    maskedImageRemovedPIL = im.fromarray(np.uint8(final_masked_blurred_image)).\
                                        convert('RGBA')
                                    brightness_blurred_imgPIL = convert_whiten_and_crop_image(brightness_blur_masked,
                                                                                                darkness)
                                    placement = (np.random.randint(0, baseImage.shape[0] * .75, 1)[0],
                                                         np.random.randint(0, baseImage.shape[1] * .75, 1)[0])
                                    maskedImageRemovedPIL.paste(brightness_blurred_imgPIL, placement,
                                                                        mask=brightness_blurred_imgPIL)
                                    final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
                                    # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add brightness and/or darkness blur to mask image
                                    idx = np.where(np.array(brightness_blurred_imgPIL)[:, :, 3] == 255)
                                    idx_x = np.minimum(placement[1] + idx[0],
                                                               np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(
                                                int)
                                    idx_y = np.minimum(placement[0] + idx[1],
                                                               np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(
                                                int)
                                    # indicator for brightness
                                    nMask[idx_x, idx_y] = 255
                            # save final image
                            saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur" \
                                    + "_a_" + str(a)  + "_s_" + str(s) + "_focus_blur_k_" + str(k) \
                                    + "_darkness_blur_al_" + str(al_d) + "_brightness_blur_al_" + str(al_b) + \
                                   args.data_extension
                            cv2.imwrite(saveName, final_masked_blurred_image)
                            # create and save ground truth mask
                            saveNameGt = args.output_data_dir + "/gt/" + baseImageName + "_gt_motion_blur" \
                                    + "_a_" + str(a)  + "_s_" + str(s) + "_focus_blur_k_" + str(k) \
                                    + "_darkness_blur_al_" + str(al_d) + "_brightness_blur_al_" + str(al_b) + \
                                     args.data_extension
                            cv2.imwrite(saveNameGt, nMask)
    # random chose the next 5 index
    for i in range(1,5):
        tempIdxs = indexs[indexs != i]
        idx = indexs[i]
        baseImage = imagesOrigonal[idx]
        baseImageName = imageNames[idx]
        for s in kernalSize:
            for a in angles:
                # motion blur
                overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                motion_blurred_overlay_img = apply_motion_blur(overlayImage, s, a)
                mask = imageOrigMask[overlayImageIdx]#saliency.compute_saliency_NN(copy.deepcopy(motion_blurred_overlay_img))
                motion_blurred_img = motion_blurred_overlay_img * mask[:, :, np.newaxis]
                #motion_blurred_img[motion_blurred_img == 0] = 255
                # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
                # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
                maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
                motion_blurred_imgPIL = convert_whiten_and_crop_image(motion_blurred_img, False)
                placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
                             np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
                maskedImageRemovedPIL.paste(motion_blurred_imgPIL, placement, mask=motion_blurred_imgPIL)
                final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]  # + motion_blurred_img
                # now save the image
                saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                cv2.imwrite(saveName, final_masked_blurred_image)
                saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                # cv2.waitKey(0)
                nMask = np.zeros(mask.shape)
                idx = np.where(np.array(motion_blurred_imgPIL)[:, :, 3] == 255)
                idx_x = np.minimum(placement[1] + idx[0], np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
                idx_y = np.minimum(placement[0] + idx[1], np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
                nMask[idx_x, idx_y] = 64
                cv2.imwrite(saveName, nMask)
                # cv2.imshow('BrightnessImage', nMask)
                # cv2.waitKey(0)
        ### out of focus blur
        for k in kernal:
            overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            focus_blur = create_out_of_focus_blur(overlayImage, k)
            #mask = saliency.compute_saliency_NN(copy.deepcopy(focus_blur))
            mask = imageOrigMask[overlayImageIdx]
            focus_blurred_img = focus_blur * mask[:, :, np.newaxis]
            #focus_blurred_img[focus_blurred_img == 0] = 255
            maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            focus_blurred_imgPIL = convert_whiten_and_crop_image(focus_blurred_img, False)
            placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
                         np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            maskedImageRemovedPIL.paste(focus_blurred_imgPIL, placement, mask=focus_blurred_imgPIL)
            final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage',final_masked_blurred_image)
            # cv2.waitKey(0)
            saveName = args.output_data_dir + "/images/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_focus_blur_k_" + str(k) + \
                       args.data_extension
            # add focus identification to mask
            nMask = np.zeros(mask.shape)
            idx = np.where(np.array(focus_blurred_imgPIL)[:, :, 3] == 255)
            idx_x = np.minimum(placement[1] + idx[0],
                               np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            idx_y = np.minimum(placement[0] + idx[1],
                               np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            nMask[idx_x, idx_y] = 128
            cv2.imwrite(saveName, nMask)
        # next we go to darkness blur
        for a in alpha_darkness:
            # create brightness/darkness blur
            overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            imageMask = imageOrigMask[overlayImageIdx]
            dark_blur = create_brightness_blur(overlayImage, a, 0)
            dark_blur_masked = dark_blur * imageMask[:, :, np.newaxis]
            dark_blur_masked[imageMask == 0] = 255
            darkness = True
            maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            dark_blurred_imgPIL = convert_whiten_and_crop_image(dark_blur_masked, darkness)
            placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
                         np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            maskedImageRemovedPIL.paste(dark_blurred_imgPIL, placement, mask=dark_blurred_imgPIL)
            final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            nMask = np.zeros(mask.shape)
            # add brightness and/or darkness blur to mask image
            idx = np.where(np.array(dark_blurred_imgPIL)[:, :, 3] == 255)
            idx_x = np.minimum(placement[1] + idx[0],
                               np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            idx_y = np.minimum(placement[0] + idx[1],
                               np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            # indicator for dark blur
            nMask[idx_x, idx_y] = 192
            # else:  # indicator for brightness
            #     nMask[idx_x, idx_y] = 255
            cv2.imwrite(saveName, nMask)
        # next we go to brightness blur
        for a in alpha_brightness:
            # create brightness blur
            overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            imageMask = imageOrigMask[overlayImageIdx]
            bright_blur = create_brightness_blur(overlayImage, a, 0)
            bright_blur_masked = bright_blur * imageMask[:, :, np.newaxis]
            darkness = False
            maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            bright_blurred_imgPIL = convert_whiten_and_crop_image(bright_blur_masked, darkness)
            placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
                         np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            maskedImageRemovedPIL.paste(bright_blurred_imgPIL, placement, mask=bright_blurred_imgPIL)
            final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            nMask = np.zeros(mask.shape)
            # add brightness and/or darkness blur to mask image
            idx = np.where(np.array(bright_blurred_imgPIL)[:, :, 3] == 255)
            idx_x = np.minimum(placement[1] + idx[0],
                               np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            idx_y = np.minimum(placement[0] + idx[1],
                               np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            # indicator for brightness
            nMask[idx_x, idx_y] = 255
            cv2.imwrite(saveName, nMask)
    # save control / no blur images
    for n in range(2):
        for i in range(len(imagesAll)):
            baseImage = imagesAll[i]
            baseImageName = imagesAllNames[i]
            saveName = args.output_data_dir + "/images/" + baseImageName + "_no_blur_" + str(n+1) + args.data_extension
            cv2.imwrite(saveName, baseImage)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_no_blur_" + str(n+1)+ args.data_extension
            nMask = np.zeros(mask.shape)
            cv2.imwrite(saveName, nMask)

def create_muri_dataset_for_training_2(args):
    # angles range for dataset
    angles = np.arange(start=0,stop=190,step=30)
    kernalSize = np.array([3, 5, 7, 9])
    # kernal
    kernal = np.array([3,5,7,9])
    # alpha
    alpha_darkness = np.arange(start=0.1, stop=0.6, step=0.1)
    alpha_brightness = np.arange(start=1.5, stop=2.0, step=0.1)
    # beta
    #beta = np.arange(start=0,stop=1,step=1) # 110 step 10
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    imagesOrigonal = []
    imageNames = []
    imageOrigMask = []
    imagesAll = []
    imagesAllNames = []
    saliency = saliency_detection.Saliency_NN()
    for imageFileName in glob.glob(args.data_dir + '/*'+ args.data_extension):
        image = cv2.imread(imageFileName)
        imageNameSplit = imageFileName.split('/')
        imageNameSplit2 = imageNameSplit[-1].split(".")
        # store all of the images and the mask of the saliency image with the image name
        imagesAll.append(image)
        # cv2.imshow('BrightnessImage', image)
        # cv2.waitKey(0)
        # https://stackoverflow.com/questions/67258207/how-can-i-join-strings-within-a-list-python
        imagesAllNames.append(''.join(imageNameSplit2[0:2]))
        if 'noblock' in imageFileName:
            continue
        imagesOrigonal.append(image)
        imageNames.append(''.join(imageNameSplit2[0:2]))
        saliency_image = saliency.compute_saliency_NN(copy.deepcopy(image))
        imageOrigMask.append(saliency_image)
        # cv2.imshow('saliency_image', image*saliency_image[:,:,np.newaxis])
        # cv2.waitKey(0)
    # now we will go through all of the images and make the dataset
    indexs = np.arange(start=0,stop=len(imagesOrigonal),step=1)
    np.random.shuffle(indexs)
    idxs = np.arange(start=0,stop=4,step=1)
    # this already makes a huge dataset
    for i in range(2):
        tempIdxs = indexs[indexs != i]
        baseImage = imagesOrigonal[i]
        baseImageName = imageNames[i]
        for s in kernalSize:
            for a in angles:
                ### out of focus blur
                for k in kernal:
                    for al_d in alpha_darkness:
                        for al_b in alpha_brightness:
                            final_masked_blurred_image = baseImage
                            nMask = np.zeros(baseImage.shape[0:2])
                            np.random.shuffle(idxs)
                            for j in idxs:
                                # motion blur
                                if j == 0:
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                                    motion_blurred_overlay_img = apply_motion_blur(overlayImage,s,a)
                                    # cv2.imshow('MotionImage1', motion_blurred_overlay_img)
                                    # cv2.waitKey(0)
                                    mask = imageOrigMask[overlayImageIdx]#saliency.compute_saliency_NN(copy.deepcopy(motion_blurred_overlay_img))
                                    motion_blurred_img = motion_blurred_overlay_img * mask[:, :, np.newaxis]
                                    #motion_blurred_img[motion_blurred_img == 0] = 255
                                    # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
                                    # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
                                    maskedImageRemovedPIL = im.fromarray(np.uint8(final_masked_blurred_image)).convert('RGBA')
                                    motion_blurred_imgPIL = convert_whiten_and_crop_image(motion_blurred_img,False)
                                    placement = (np.random.randint(0,baseImage.shape[0]*.75,1)[0],
                                             np.random.randint(0,baseImage.shape[1]*.75,1)[0])
                                    maskedImageRemovedPIL.paste(motion_blurred_imgPIL, placement,mask=motion_blurred_imgPIL)
                                    final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:,:,0:3]
                                    # cv2.imshow('MotionImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    idx = np.where(np.array(motion_blurred_imgPIL)[:,:,3] == 255)
                                    idx_x = np.minimum(placement[1] + idx[0],
                                                   np.ones(idx[0].shape[0]) * (nMask.shape[0]-1)).astype(int)
                                    idx_y = np.minimum(placement[0] + idx[1],
                                                    np.ones(idx[1].shape[0]) * (nMask.shape[1]-1)).astype(int)
                                    nMask[idx_x, idx_y] = 64
                                    # cv2.imshow('BrightnessImage', nMask)
                                    # cv2.waitKey(0)
                                # focus blur
                                elif j == 1:
                                    # now do out of focus blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                                    focus_blur = create_out_of_focus_blur(overlayImage, k)
                                    mask_focus = imageOrigMask[overlayImageIdx]#saliency.compute_saliency_NN(copy.deepcopy(focus_blur))
                                    focus_blurred_img = focus_blur * mask_focus[:, :, np.newaxis]
                                    #focus_blurred_img[focus_blurred_img == 0] = 255
                                    maskedImageRemovedPIL = im.fromarray(np.uint8(final_masked_blurred_image)).convert('RGBA')
                                    focus_blurred_imgPIL = convert_whiten_and_crop_image(focus_blurred_img,False)
                                    placement = (np.random.randint(0, baseImage.shape[0]*.75, 1)[0],
                                                np.random.randint(0, baseImage.shape[1]*.75, 1)[0])
                                    maskedImageRemovedPIL.paste(focus_blurred_imgPIL, placement,mask=focus_blurred_imgPIL)
                                    final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
                                    # cv2.imshow('FocusImage',final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add focus identification to mask
                                    idx = np.where(np.array(focus_blurred_imgPIL)[:, :, 3] == 255)
                                    idx_x = np.minimum(placement[1] + idx[0],
                                                    np.ones(idx[0].shape[0])*(nMask.shape[0]-1)).astype(int)
                                    idx_y = np.minimum(placement[0] + idx[1],
                                                    np.ones(idx[1].shape[0])*(nMask.shape[1]-1)).astype(int)
                                    nMask[idx_x, idx_y] = 128
                                # darkness blur
                                elif j == 2:
                                    # create darkness blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                                    imageMask = imageOrigMask[overlayImageIdx]
                                    dark_blur = create_brightness_blur(overlayImage, al_d,0)
                                    # cv2.imshow('BrightnessImage', dark_blur)
                                    # cv2.waitKey(0)
                                    dark_blur_masked = dark_blur * imageMask[:, :, np.newaxis]
                                    dark_blur_masked[imageMask == 0] = 255
                                    # cv2.imshow('BrightnessImage_mask', dark_blur_masked)
                                    # cv2.waitKey(0)
                                    darkness = True
                                    maskedImageRemovedPIL = im.fromarray(np.uint8(final_masked_blurred_image)).\
                                            convert('RGBA')
                                    dark_blurred_imgPIL = convert_whiten_and_crop_image(dark_blur_masked,darkness)
                                    # cv2.imshow('BrightnessImage2', np.array(dark_blurred_imgPIL)[:,:,0:3])
                                    # cv2.waitKey(0)
                                    placement = (np.random.randint(0, baseImage.shape[0]*.75, 1)[0],
                                                np.random.randint(0, baseImage.shape[1]*.75, 1)[0])
                                    maskedImageRemovedPIL.paste(dark_blurred_imgPIL, placement,
                                                                    mask=dark_blurred_imgPIL)
                                    final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
                                    # cv2.imshow('DarknessImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add brightness and/or darkness blur to mask image
                                    idx = np.where(np.array(dark_blurred_imgPIL)[:, :, 3] == 255)
                                    idx_x = np.minimum(placement[1] + idx[0],
                                                    np.ones(idx[0].shape[0]) * (nMask.shape[0]-1)).astype(int)
                                    idx_y = np.minimum(placement[0] + idx[1],
                                                    np.ones(idx[1].shape[0]) * (nMask.shape[1]-1)).astype(int)
                                    # indicator for dark blur
                                    nMask[idx_x, idx_y] = 192
                                # brightness blur
                                elif j == 3:
                                    # create brightness blur
                                    overlayImageIdx = np.random.choice(tempIdxs, 1, replace=False)[0]
                                    overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                                    imageMask = imageOrigMask[overlayImageIdx]
                                    brightness_blur = create_brightness_blur(overlayImage, al_b,0)
                                    brightness_blur_masked = brightness_blur * imageMask[:, :, np.newaxis]
                                    darkness = False
                                    maskedImageRemovedPIL = im.fromarray(np.uint8(final_masked_blurred_image)).\
                                        convert('RGBA')
                                    brightness_blurred_imgPIL = convert_whiten_and_crop_image(brightness_blur_masked,
                                                                                                darkness)
                                    placement = (np.random.randint(0, baseImage.shape[0] * .75, 1)[0],
                                                         np.random.randint(0, baseImage.shape[1] * .75, 1)[0])
                                    maskedImageRemovedPIL.paste(brightness_blurred_imgPIL, placement,
                                                                        mask=brightness_blurred_imgPIL)
                                    final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
                                    # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                                    # cv2.waitKey(0)
                                    # add brightness and/or darkness blur to mask image
                                    idx = np.where(np.array(brightness_blurred_imgPIL)[:, :, 3] == 255)
                                    idx_x = np.minimum(placement[1] + idx[0],
                                                               np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(
                                                int)
                                    idx_y = np.minimum(placement[0] + idx[1],
                                                               np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(
                                                int)
                                    # indicator for brightness
                                    nMask[idx_x, idx_y] = 255
                            # save final image
                            saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur" \
                                    + "_a_" + str(a)  + "_s_" + str(s) + "_focus_blur_k_" + str(k) \
                                    + "_darkness_blur_al_" + str(al_d) + "_brightness_blur_al_" + str(al_b) + \
                                   args.data_extension
                            cv2.imwrite(saveName, final_masked_blurred_image)
                            # create and save ground truth mask
                            saveNameGt = args.output_data_dir + "/gt/" + baseImageName + "_gt_motion_blur" \
                                    + "_a_" + str(a)  + "_s_" + str(s) + "_focus_blur_k_" + str(k) \
                                    + "_darkness_blur_al_" + str(al_d) + "_brightness_blur_al_" + str(al_b) + \
                                     args.data_extension
                            cv2.imwrite(saveNameGt, nMask)
    # random chose the next 5 index
    for i in range(2,len(imagesOrigonal)):
        #tempIdxs = indexs[indexs != i]
        idx = indexs[i]
        final_masked_blurred_image = imagesOrigonal[idx]
        baseImageName = imageNames[idx]
        for s in kernalSize:
            for a in angles:
                # motion blur
                overlayImageIdx = idx#np.random.choice(tempIdxs, 1, replace=False)[0]
                overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                motion_blurred_overlay_img = apply_motion_blur(overlayImage, s, a)
                mask = imageOrigMask[overlayImageIdx]#saliency.compute_saliency_NN(copy.deepcopy(motion_blurred_overlay_img))
                motion_blurred_img = motion_blurred_overlay_img * mask[:, :, np.newaxis]
                maskedImageRemoved = final_masked_blurred_image * np.logical_not(mask[:, :, np.newaxis])
                final_masked_blurred_image = maskedImageRemoved + motion_blurred_img
                # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
                # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap

                #maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
                #motion_blurred_imgPIL = convert_whiten_and_crop_image(motion_blurred_img, False)
                #placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
                #             np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
                #maskedImageRemovedPIL.paste(motion_blurred_imgPIL, placement, mask=motion_blurred_imgPIL)
                #final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]  # + motion_blurred_img
                # now save the image
                saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                cv2.imwrite(saveName, final_masked_blurred_image)
                saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                # cv2.waitKey(0)
                nMask = np.zeros(mask.shape)
                nMask[mask == 1] = 64
                # idx = np.where(np.array(motion_blurred_imgPIL)[:, :, 3] == 255)
                # idx_x = np.minimum(placement[1] + idx[0], np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
                # idx_y = np.minimum(placement[0] + idx[1], np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
                # nMask[idx_x, idx_y] = 64
                cv2.imwrite(saveName, nMask)
                # cv2.imshow('BrightnessImage', nMask)
                # cv2.waitKey(0)
        ### out of focus blur
        for k in kernal:
            overlayImageIdx = idx#np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            focus_blur = create_out_of_focus_blur(overlayImage, k)
            mask = imageOrigMask[overlayImageIdx]
            focus_blurred_img = focus_blur * mask[:, :, np.newaxis]
            maskedImageRemoved = final_masked_blurred_image * np.logical_not(mask[:, :, np.newaxis])
            final_masked_blurred_image = maskedImageRemoved + focus_blurred_img
            # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            # focus_blurred_imgPIL = convert_whiten_and_crop_image(focus_blurred_img, False)
            # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
            #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            # maskedImageRemovedPIL.paste(focus_blurred_imgPIL, placement, mask=focus_blurred_imgPIL)
            # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage',final_masked_blurred_image)
            # cv2.waitKey(0)
            saveName = args.output_data_dir + "/images/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_focus_blur_k_" + str(k) + \
                       args.data_extension
            # add focus identification to mask
            nMask = np.zeros(mask.shape)
            nMask[mask == 1] = 128
            # idx = np.where(np.array(focus_blurred_imgPIL)[:, :, 3] == 255)
            # idx_x = np.minimum(placement[1] + idx[0],
            #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            # idx_y = np.minimum(placement[0] + idx[1],
            #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            # nMask[idx_x, idx_y] = 128
            cv2.imwrite(saveName, nMask)
        # next we go to darkness blur
        for a in alpha_darkness:
            # create brightness/darkness blur
            overlayImageIdx = idx#np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            imageMask = imageOrigMask[overlayImageIdx]
            dark_blur = create_brightness_blur(overlayImage, a, 0)
            dark_blur_masked = dark_blur * imageMask[:, :, np.newaxis]
            maskedImageRemoved = final_masked_blurred_image * np.logical_not(mask[:, :, np.newaxis])
            final_masked_blurred_image = maskedImageRemoved + dark_blur_masked
            # dark_blur_masked[imageMask == 0] = 255
            # darkness = True
            # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            # dark_blurred_imgPIL = convert_whiten_and_crop_image(dark_blur_masked, darkness)
            # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
            #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            # maskedImageRemovedPIL.paste(dark_blurred_imgPIL, placement, mask=dark_blurred_imgPIL)
            # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            nMask = np.zeros(mask.shape)
            nMask[mask == 1] = 192
            # add brightness and/or darkness blur to mask image
            # idx = np.where(np.array(dark_blurred_imgPIL)[:, :, 3] == 255)
            # idx_x = np.minimum(placement[1] + idx[0],
            #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            # idx_y = np.minimum(placement[0] + idx[1],
            #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            # indicator for dark blur
            # nMask[idx_x, idx_y] = 192
            # else:  # indicator for brightness
            #     nMask[idx_x, idx_y] = 255
            cv2.imwrite(saveName, nMask)
        # next we go to brightness blur
        for a in alpha_brightness:
            # create brightness blur
            overlayImageIdx = idx#np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            imageMask = imageOrigMask[overlayImageIdx]
            bright_blur = create_brightness_blur(overlayImage, a, 0)
            bright_blur_masked = bright_blur * imageMask[:, :, np.newaxis]
            maskedImageRemoved = final_masked_blurred_image * np.logical_not(mask[:, :, np.newaxis])
            final_masked_blurred_image = maskedImageRemoved + bright_blur_masked
            # darkness = False
            # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            # bright_blurred_imgPIL = convert_whiten_and_crop_image(bright_blur_masked, darkness)
            # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
            #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            # maskedImageRemovedPIL.paste(bright_blurred_imgPIL, placement, mask=bright_blurred_imgPIL)
            # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            nMask = np.zeros(mask.shape)
            nMask[mask == 1] = 255
            # add brightness and/or darkness blur to mask image
            # idx = np.where(np.array(bright_blurred_imgPIL)[:, :, 3] == 255)
            # idx_x = np.minimum(placement[1] + idx[0],
            #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            # idx_y = np.minimum(placement[0] + idx[1],
            #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            # indicator for brightness
            # nMask[idx_x, idx_y] = 255
            cv2.imwrite(saveName, nMask)
    # save control / no blur images
    for n in range(3):
        for i in range(len(imagesAll)):
            baseImage = imagesAll[i]
            baseImageName = imagesAllNames[i]
            saveName = args.output_data_dir + "/images/" + baseImageName + "_no_blur_" + str(n+1) + args.data_extension
            cv2.imwrite(saveName, baseImage)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_no_blur_" + str(n+1)+ args.data_extension
            nMask = np.zeros(mask.shape)
            cv2.imwrite(saveName, nMask)

def create_muri_dataset_for_testing_and_validation(args):
    # angles range for dataset
    angles = np.arange(start=0,stop=190,step=30)
    kernalSize = np.array([3, 5, 7, 9])
    # kernal
    kernal = np.array([3,5,7,9])
    # alpha
    alpha_darkness = np.arange(start=0.1, stop=0.6, step=0.1)
    alpha_brightness = np.arange(start=1.5, stop=2.0, step=0.1)
    # beta
    #beta = np.arange(start=0,stop=1,step=1) # 110 step 10
    tl.files.exists_or_mkdir(args.output_data_dir + "/images/")
    tl.files.exists_or_mkdir(args.output_data_dir + "/gt/")
    # list of all original Images
    imagesOrigonal = []
    imageNames = []
    imageOrigMask = []
    imagesAll = []
    imagesAllNames = []
    saliency = saliency_detection.Saliency_NN()
    for imageFileName in glob.glob(args.data_dir + '/*'+ args.data_extension):
        image = cv2.imread(imageFileName)
        imageNameSplit = imageFileName.split('/')
        imageNameSplit2 = imageNameSplit[-1].split(".")
        # store all of the images and the mask of the saliency image with the image name
        imagesAll.append(image)
        # cv2.imshow('BrightnessImage', image)
        # cv2.waitKey(0)
        # https://stackoverflow.com/questions/67258207/how-can-i-join-strings-within-a-list-python
        imagesAllNames.append(''.join(imageNameSplit2[0:2]))
        if 'noblock' in imageFileName:
            continue
        imagesOrigonal.append(image)
        imageNames.append(''.join(imageNameSplit2[0:2]))
        saliency_image = saliency.compute_saliency_NN(copy.deepcopy(image))
        imageOrigMask.append(saliency_image)
        # cv2.imshow('saliency_image', image*saliency_image[:,:,np.newaxis])
        # cv2.waitKey(0)
    # now we will go through all of the images and make the dataset
    indexs = np.arange(start=0,stop=len(imagesOrigonal),step=1)
    np.random.shuffle(indexs)
    for i in range(len(imagesOrigonal)):
        #tempIdxs = indexs[indexs != i]
        idx = indexs[i]
        final_masked_blurred_image = imagesOrigonal[idx]
        baseImageName = imageNames[idx]
        for s in kernalSize:
            for a in angles:
                # motion blur
                overlayImageIdx = idx  # np.random.choice(tempIdxs, 1, replace=False)[0]
                overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
                motion_blurred_overlay_img = apply_motion_blur(overlayImage, s, a)
                mask = imageOrigMask[
                    overlayImageIdx]  # saliency.compute_saliency_NN(copy.deepcopy(motion_blurred_overlay_img))
                motion_blurred_img = motion_blurred_overlay_img * mask[:, :, np.newaxis]
                maskedImageRemoved = final_masked_blurred_image * np.logical_not(mask[:, :, np.newaxis])
                final_masked_blurred_image = maskedImageRemoved + motion_blurred_img
                # https://stackoverflow.com/questions/31273592/valueerror-bad-transparency-mask-when-pasting-one-image-onto-another-with-pyt
                # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap

                # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
                # motion_blurred_imgPIL = convert_whiten_and_crop_image(motion_blurred_img, False)
                # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
                #             np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
                # maskedImageRemovedPIL.paste(motion_blurred_imgPIL, placement, mask=motion_blurred_imgPIL)
                # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]  # + motion_blurred_img
                # now save the image
                saveName = args.output_data_dir + "/images/" + baseImageName + "_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                cv2.imwrite(saveName, final_masked_blurred_image)
                saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_motion_blur_a_" + str(a) + "_s_" + \
                           str(s) + args.data_extension
                # cv2.imshow('BrightnessImage', final_masked_blurred_image)
                # cv2.waitKey(0)
                nMask = np.zeros(mask.shape)
                nMask[mask == 1] = 64
                # idx = np.where(np.array(motion_blurred_imgPIL)[:, :, 3] == 255)
                # idx_x = np.minimum(placement[1] + idx[0], np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
                # idx_y = np.minimum(placement[0] + idx[1], np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
                # nMask[idx_x, idx_y] = 64
                cv2.imwrite(saveName, nMask)
                # cv2.imshow('BrightnessImage', nMask)
                # cv2.waitKey(0)
        ### out of focus blur
        for k in kernal:
            overlayImageIdx = idx  # np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            focus_blur = create_out_of_focus_blur(overlayImage, k)
            mask = imageOrigMask[overlayImageIdx]
            focus_blurred_img = focus_blur * mask[:, :, np.newaxis]
            maskedImageRemoved = final_masked_blurred_image * np.logical_not(mask[:, :, np.newaxis])
            final_masked_blurred_image = maskedImageRemoved + focus_blurred_img
            # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            # focus_blurred_imgPIL = convert_whiten_and_crop_image(focus_blurred_img, False)
            # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
            #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            # maskedImageRemovedPIL.paste(focus_blurred_imgPIL, placement, mask=focus_blurred_imgPIL)
            # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage',final_masked_blurred_image)
            # cv2.waitKey(0)
            saveName = args.output_data_dir + "/images/" + baseImageName + "_focus_blur_k_" + str(k) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_focus_blur_k_" + str(k) + \
                       args.data_extension
            # add focus identification to mask
            nMask = np.zeros(mask.shape)
            nMask[mask == 1] = 128
            # idx = np.where(np.array(focus_blurred_imgPIL)[:, :, 3] == 255)
            # idx_x = np.minimum(placement[1] + idx[0],
            #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            # idx_y = np.minimum(placement[0] + idx[1],
            #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            # nMask[idx_x, idx_y] = 128
            cv2.imwrite(saveName, nMask)
        # next we go to darkness blur
        for a in alpha_darkness:
            # create brightness/darkness blur
            overlayImageIdx = idx  # np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            imageMask = imageOrigMask[overlayImageIdx]
            dark_blur = create_brightness_blur(overlayImage, a, 0)
            dark_blur_masked = dark_blur * imageMask[:, :, np.newaxis]
            maskedImageRemoved = final_masked_blurred_image * np.logical_not(mask[:, :, np.newaxis])
            final_masked_blurred_image = maskedImageRemoved + dark_blur_masked
            # dark_blur_masked[imageMask == 0] = 255
            # darkness = True
            # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            # dark_blurred_imgPIL = convert_whiten_and_crop_image(dark_blur_masked, darkness)
            # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
            #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            # maskedImageRemovedPIL.paste(dark_blurred_imgPIL, placement, mask=dark_blurred_imgPIL)
            # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_darkness_blur_al_" + str(a) + \
                       args.data_extension
            nMask = np.zeros(mask.shape)
            nMask[mask == 1] = 192
            # add brightness and/or darkness blur to mask image
            # idx = np.where(np.array(dark_blurred_imgPIL)[:, :, 3] == 255)
            # idx_x = np.minimum(placement[1] + idx[0],
            #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            # idx_y = np.minimum(placement[0] + idx[1],
            #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            # indicator for dark blur
            # nMask[idx_x, idx_y] = 192
            # else:  # indicator for brightness
            #     nMask[idx_x, idx_y] = 255
            cv2.imwrite(saveName, nMask)
        # next we go to brightness blur
        for a in alpha_brightness:
            # create brightness blur
            overlayImageIdx = idx  # np.random.choice(tempIdxs, 1, replace=False)[0]
            overlayImage = copy.deepcopy(imagesOrigonal[overlayImageIdx])
            imageMask = imageOrigMask[overlayImageIdx]
            bright_blur = create_brightness_blur(overlayImage, a, 0)
            bright_blur_masked = bright_blur * imageMask[:, :, np.newaxis]
            maskedImageRemoved = final_masked_blurred_image * np.logical_not(mask[:, :, np.newaxis])
            final_masked_blurred_image = maskedImageRemoved + bright_blur_masked
            # darkness = False
            # maskedImageRemovedPIL = im.fromarray(np.uint8(baseImage)).convert('RGBA')
            # bright_blurred_imgPIL = convert_whiten_and_crop_image(bright_blur_masked, darkness)
            # placement = (np.random.randint(0, baseImage.shape[0] * .50, 1)[0],
            #              np.random.randint(0, baseImage.shape[1] * .50, 1)[0])
            # maskedImageRemovedPIL.paste(bright_blurred_imgPIL, placement, mask=bright_blurred_imgPIL)
            # final_masked_blurred_image = np.array(maskedImageRemovedPIL)[:, :, 0:3]
            # cv2.imshow('BrightnessImage', final_masked_blurred_image)
            # cv2.waitKey(0)
            # save the image
            saveName = args.output_data_dir + "/images/" + baseImageName + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            cv2.imwrite(saveName, final_masked_blurred_image)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_brightness_blur_al_" + str(a) + \
                       args.data_extension
            nMask = np.zeros(mask.shape)
            nMask[mask == 1] = 255
            # add brightness and/or darkness blur to mask image
            # idx = np.where(np.array(bright_blurred_imgPIL)[:, :, 3] == 255)
            # idx_x = np.minimum(placement[1] + idx[0],
            #                    np.ones(idx[0].shape[0]) * (nMask.shape[0] - 1)).astype(int)
            # idx_y = np.minimum(placement[0] + idx[1],
            #                    np.ones(idx[1].shape[0]) * (nMask.shape[1] - 1)).astype(int)
            # indicator for brightness
            # nMask[idx_x, idx_y] = 255
            cv2.imwrite(saveName, nMask)
    # save control / no blur images
    for n in range(2):
        for i in range(len(imagesAll)):
            baseImage = imagesAll[i]
            baseImageName = imagesAllNames[i]
            saveName = args.output_data_dir + "/images/" + baseImageName + "_no_blur_" + str(n+1) + args.data_extension
            cv2.imwrite(saveName, baseImage)
            # get ground truth mask
            saveName = args.output_data_dir + "/gt/" + baseImageName + "_gt_" + "_no_blur_" + str(n+1)+ args.data_extension
            nMask = np.zeros(mask.shape)
            cv2.imwrite(saveName, nMask)

if __name__ == "__main__":
 # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS MURI CREATE BLUR DATASET')
    # directory data location
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_data_dir', type=str)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str,default=".png")
    parser.add_argument('--is_testing', default=False, action='store_true')
    args = parser.parse_args()
    if args.is_testing:
        create_muri_dataset_for_testing_and_validation(args)
    else:
        create_muri_dataset_for_training_2(args)


