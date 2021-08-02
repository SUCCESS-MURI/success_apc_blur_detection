# Author mhatfalv
import argparse
import copy
import glob
import os

import cv2
import numpy as np
from numba import jit, prange
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from PIL import Image as im
# used maryal version of saliency map decomposition and feature extraction
double_center_surround = True
pyramid_levels = 2  # default note cannot be 1
divFactor = 1

threshold_dict = {
            'red': {
                # 'L': [6.293,  82.456],
                # 'A': [25.368, 62.619],
                # 'B': [-32.842, 54.350]
                'L': [0.0,  35.714],
                'A': [32.044, 80.092],
                'B': [-10.987, 33.918]
            },
            'blue':{
                # 'L': [31.081, 68.958],
                # 'A': [-35.972, 8.049],
                # 'B': [-39.403, -2.443]
                'L': [23.892, 50.985],
                'A': [-12.205, 5.992],
                'B': [-38.495, -11.859]
            },
            'green':{
                'L': [3.160, 82.431],
                'A': [-38.928, -9.685],
                'B': [-1.965, 57.163]
            },
            # 'yellow':{
            #     'L': [33.499, 81.637],
            #     'A': [-28.605, 44.670],
            #     'B': [43.439, 69.638]
            # },
            'violet':  {
                # 'L': [3.302, 82.354],
                # 'A': [15.743, 46.923],
                # 'B': [-46.238, -8.336]
                'L': [14.163, 53.202],
                'A': [16.502, 32.309],
                'B': [-39.559, -10.417]
            }
        }

threshold_dict_hsv = {
            'red': {
                'H': [151,  180],
                'S': [50, 255],
                'V': [50, 255]
            },
            'blue':{
                'H': [100,  119],
                'S': [100, 255],
                'V': [100, 255]
            },
            'green':{
                'H': [46, 75],
                'S': [100, 255],
                'V': [100, 255]
            },
            'violet':  {
                'H': [120,  150],
                'S': [50, 255],
                'V': [50, 255]
            },
            'yellow':{
                'H': [10,  45],
                'S': [50, 255],
                'V': [50, 255]
            }
        }

# decompose rgb
def decompose_rgba(image):
    red_image = image[:, :, 2]
    green_image = image[:, :, 1]
    blue_image = image[:, :, 0]
    return red_image, green_image, blue_image


def decompose_rgb_into_hsv(image):
    imagehsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_image = imagehsv[:, :, 0]
    s_image = imagehsv[:, :, 1]
    v_image = imagehsv[:, :, 2]
    return h_image.astype(float), s_image.astype(float), v_image.astype(float)


# compute intensity of image and scale by mean
def get_intensity(r, g, b):
    intensity = r + g + b
    intensity = np.divide(intensity.astype(float), 3.0)
    return intensity


# cross-scale center-surround operator (inverse)
def compute_center_surround(bigger_img, smaller_img):
    height, width = bigger_img.shape
    sheight, swidth = smaller_img.shape
    enarged_sImg = np.empty([height, width])
    # enlarge the smaller image using bilinear interpolation
    enarged_sImg = compute_bilinear_interpolation(smaller_img, swidth, sheight, enarged_sImg, width, height)
    scaled = bigger_img - enarged_sImg
    if double_center_surround:
        scaled = np.maximum(scaled, np.zeros(scaled.shape))
    else:
        scaled = np.absolute(scaled)
    # return max normalized array
    return max_normalize(scaled)


# Normalize image depending on number of local maximums
@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def max_normalize(image):
    threshold = np.max(image)
    threshold = threshold * 0.5
    height, width = image.shape
    # img = copy.deepcopy(image)
    img = image
    m = 0.0
    for i in prange(1, height - 1):
        for j in prange(1, width - 1):
            tmp = max(img[i - 1, j], img[i + 1, j])
            tmp = max(img[i, j], tmp)
            tmp1 = max(img[i, j - 1], img[i, j + 1])
            tmp1 = max(tmp1, tmp)
            for k in prange(i - 1, i + 2):
                for l in prange(j - 1, j + 2):
                    if threshold < img[k, l] == tmp1:
                        m = m + 1.0
    if m > 0:
        m = 1.0 / np.sqrt(m)
        img = img * m
    return img


# interpolate bilinear
# code from
# https://eng.aurelienpierre.com/2020/03/bilinear-interpolation-on-images-stored-as-python-numpy-ndarray/
@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def compute_bilinear_interpolation(array_in, width_in, height_in, array_out, width_out, height_out):
    for i in prange(height_out):
        for j in prange(width_out):
            # Relative coordinates of the pixel in output space
            x_out = j / width_out
            y_out = i / height_out

            # Corresponding absolute coordinates of the pixel in input space
            x_in = (x_out * width_in)
            y_in = (y_out * height_in)

            # Nearest neighbours coordinates in input space
            x_prev = int(np.floor(x_in))
            x_next = x_prev + 1
            y_prev = int(np.floor(y_in))
            y_next = y_prev + 1

            # Sanitize bounds - no need to check for < 0
            x_prev = min(x_prev, width_in - 1)
            x_next = min(x_next, width_in - 1)
            y_prev = min(y_prev, height_in - 1)
            y_next = min(y_next, height_in - 1)

            # Distances between neighbour nodes in input space
            Dy_next = y_next - y_in;
            Dy_prev = 1. - Dy_next;  # because next - prev = 1
            Dx_next = x_next - x_in;
            Dx_prev = 1. - Dx_next;  # because next - prev = 1

            # Interpolate over 1 layer
            array_out[i][j] = Dy_prev * (array_in[y_next][x_prev] * Dx_next + array_in[y_next][x_next] * Dx_prev) \
                              + Dy_next * (array_in[y_prev][x_prev] * Dx_next + array_in[y_prev][x_next] * Dx_prev)
    return array_out


# shrink the image by half
# using gaussian blur and downscaling
def compute_pyramid(image, numofLayers):
    # copy the image
    im = copy.deepcopy(image)
    images = []
    # for now don't return scale
    scale = []
    scale.append(1.0)
    images.append(im)
    for i in range(numofLayers):
        # pyr down scales by half each time
        im = cv2.pyrDown(im)
        images.append(im)
        scale.append(scale[i] / 2.0)
    return images


def get_features_itti(image):
    # first lets get the rgb features
    r, g, b = decompose_rgba(image)
    # Compute color features
    # need to conver to floats
    featRG, featBY = compute_color_features(r.astype(float), g.astype(float), b.astype(float))
    featInt = get_intensity(r, g, b)
    return featRG, featBY, featInt


# code based off of maryal see_opponency
# C++ version on
# https://github.com/marynelv/assisted-photography/blob/master/frameworks/src/Framework-See/See/ImageConversion.cpp
def compute_color_features(r, g, b):
    # max(b,max(r,g))
    ma = np.maximum(b, np.maximum(r, g))
    # get red green feature
    rg = r - g
    # get blue yellow feature
    mi = np.minimum(r, g)
    by = b - mi
    # set zeros at low luminance
    rg[ma < 25] = 0.0
    by[ma < 25] = 0.0
    # now scale
    rg[ma >= 25] = rg[ma >= 25] / ma[ma >= 25]
    by[ma >= 25] = by[ma >= 25] / ma[ma >= 25]
    return rg, by


def compute_saliency_Itti(args):
    sailencyMaps = []
    for imageFileName in glob.glob(args.data_dir + '/Incorrect/*' + args.data_extension):
        image = cv2.imread(imageFileName)
        featRG, featBY, featInt = get_features_itti(image.astype(float))
        featH, featS, featV = decompose_rgb_into_hsv(image)
        # # view features
        # # show the output image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.imshow("Image Feature Red-Green", featRG)
        # cv2.waitKey(0)
        # cv2.imshow("Image Feature Blue-Yellow", featBY)
        # cv2.waitKey(0)
        # cv2.imshow("Image Feature Intensity", featInt)
        # cv2.waitKey(0)
        # compute pyramid of each feature image
        pyrFeatRG = compute_pyramid(featRG, pyramid_levels)
        pyrFeatBY = compute_pyramid(featBY, pyramid_levels)
        pyrFeatInt = compute_pyramid(featInt, pyramid_levels)
        pyrFeatH = compute_pyramid(featH, pyramid_levels)
        pyrFeatS = compute_pyramid(featS, pyramid_levels)
        pyrFeatV = compute_pyramid(featV, pyramid_levels)
        pyrSurrH = []
        pyrSurrS = []
        pyrSurrV = []
        pyrSurrInt = []
        pyrSurrRG = []
        pyrSurrBY = []
        height, width, _ = image.shape
        for level in range(pyramid_levels - 1):
            # intensity
            surround = compute_center_surround(pyrFeatInt[level], pyrFeatInt[level + 1])
            surround_enlarge = np.empty([height, width])
            surround_enlarge = compute_bilinear_interpolation(surround, surround.shape[0], surround.shape[1],
                                                              surround_enlarge, width, height)
            pyrSurrInt.append(surround_enlarge)
            # r-g
            surround = compute_center_surround(pyrFeatRG[level], pyrFeatRG[level + 1])
            surround_enlarge = np.empty([height, width])
            surround_enlarge = compute_bilinear_interpolation(surround, surround.shape[0], surround.shape[1],
                                                              surround_enlarge, width, height)
            pyrSurrRG.append(surround_enlarge)
            # b-y
            surround = compute_center_surround(pyrFeatBY[level], pyrFeatBY[level + 1])
            surround_enlarge = np.empty([height, width])
            surround_enlarge = compute_bilinear_interpolation(surround, surround.shape[0], surround.shape[1],
                                                              surround_enlarge, width, height)
            pyrSurrBY.append(surround_enlarge)
            # h
            surround = compute_center_surround(pyrFeatH[level], pyrFeatBY[level + 1])
            surround_enlarge = np.empty([height, width])
            surround_enlarge = compute_bilinear_interpolation(surround, surround.shape[0], surround.shape[1],
                                                              surround_enlarge, width, height)
            pyrSurrH.append(surround_enlarge)
            # s
            surround = compute_center_surround(pyrFeatS[level], pyrFeatBY[level + 1])
            surround_enlarge = np.empty([height, width])
            surround_enlarge = compute_bilinear_interpolation(surround, surround.shape[0], surround.shape[1],
                                                              surround_enlarge, width, height)
            pyrSurrS.append(surround_enlarge)
            # v
            surround = compute_center_surround(pyrFeatV[level], pyrFeatBY[level + 1])
            surround_enlarge = np.empty([height, width])
            surround_enlarge = compute_bilinear_interpolation(surround, surround.shape[0], surround.shape[1],
                                                              surround_enlarge, width, height)
            pyrSurrV.append(surround_enlarge)

        # need to add pyramid levels
        for level in range(1, pyramid_levels - 1):
            pyrSurrInt[0] += pyrSurrInt[level]
            pyrSurrRG[0] += pyrSurrRG[level]
            pyrSurrBY[0] += pyrSurrBY[level]
            pyrSurrH[0] += pyrSurrH[level]
            pyrSurrS[0] += pyrSurrS[level]
            pyrSurrV[0] += pyrSurrV[level]
        intNorm = max_normalize(pyrSurrInt[0])
        rgNorm = max_normalize(pyrSurrRG[0])
        byNorm = max_normalize(pyrSurrBY[0])
        hNorm = max_normalize(pyrSurrH[0])
        sNorm = max_normalize(pyrSurrS[0])
        vNorm = max_normalize(pyrSurrV[0])
        # normal maryel method
        # NormRGBY = rgNorm + byNorm
        # NormRGBY = max_normalize(NormRGBY)
        # NormRGBYInt = NormRGBY + intNorm
        # saliency = NormRGBYInt * divFactor
        saliency = (rgNorm + byNorm + intNorm + hNorm + sNorm + vNorm) * divFactor
        # view features
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.imshow("Image Saliency", saliency)
        cv2.waitKey(0)
        cv2.normalize(saliency, saliency, 0, 255, cv2.NORM_MINMAX)
        saliencies = np.uint8([saliency, saliency, saliency])
        saliency = cv2.merge(saliencies)
        # mean shift filtering
        cv2.pyrMeanShiftFiltering(saliency, 20, 200, saliency, 2)
        # change to grayscale
        saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
        # then equalize histogram
        cv2.equalizeHist(saliency, saliency)
        (T, saliency) = cv2.threshold(225 - saliency, 200, 255, cv2.THRESH_BINARY)
        mask = refine_saliency_with_grabcut(image, saliency)
        segmentation = image * mask[:, :, np.newaxis]
        # view saliency
        cv2.imshow("Image Saliency Threshold", segmentation)
        cv2.waitKey(0)
        sailencyMaps.append(segmentation)
    return sailencyMaps


# following backprop code from
# https://github.com/jacobgil/saliency-from-backproj/blob/master/saliency.py
# written by Jacob Gildenblat
# jacobgil
####################################################################
def compute_backpropagation(source, target, levels=2, scale=1):
    # convert channels from BGR to HSV
    hsv_source = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # calculate object histogram and return region of interset (roi)
    roi_source_hist = cv2.calcHist([hsv_source], [0, 1], None, [levels, levels], [0, 180, 0, 256])
    # normalize histogram and apply backprojection
    cv2.normalize(roi_source_hist, roi_source_hist, 0, 255, cv2.NORM_MINMAX)
    # BackProjection is the probability that a pixel in target Image
    # belongs to a roi based on the model histogram that we computed using roi_source_hist
    backprop_image = cv2.calcBackProject([hsv_target], [0, 1], roi_source_hist, [0, 180, 0, 256], scale)
    # cv2.imshow("Image Saliency Threshold", backprop_image)
    # cv2.waitKey(0)
    return backprop_image


def compute_saliency_by_backprojection(img):
    # mean shift filtering
    cv2.pyrMeanShiftFiltering(img, 2, 10, img, 4)
    # compute backpropagation
    backproj = np.uint8(compute_backpropagation(img, img, levels=2))
    # normalize and format
    cv2.normalize(backproj, backproj, 0, 255, cv2.NORM_MINMAX)
    saliencies = [backproj, backproj, backproj]
    saliency = cv2.merge(saliencies)
    # mean shift filtering
    cv2.pyrMeanShiftFiltering(saliency, 20, 200, saliency, 2)
    # change to grayscale
    saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
    # then equalize histogram
    cv2.equalizeHist(saliency, saliency)
    # return opposite of color value
    (T, saliency) = cv2.threshold(255-saliency, 200, 255, cv2.THRESH_BINARY)
    return saliency

# compute the contours
def largest_contours_rect(saliency):
    contours, hierarchy = cv2.findContours(saliency, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours by smallest to largest area
    contours = sorted(contours, key=cv2.contourArea)
    # return largest area
    return cv2.boundingRect(contours[-1])

# grabcut finds the contors and returns the masked image
def refine_saliency_with_grabcut(img, saliency):
    rect = largest_contours_rect(saliency)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    saliency[np.where(saliency > 0)] = cv2.GC_FGD
    mask = saliency
    cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask

# computes the saliency by backpropagation method for one image
def backprojection_saliency(img):
    saliency = compute_saliency_by_backprojection(img)
    mask = refine_saliency_with_grabcut(img, saliency)
    return mask

##########################################################################3

# test on all images and return sailency maps
def compute_saliency_backprop(args):
    sailencyMaps = []
    for imageFileName in glob.glob('/home/mary/code/local_success_dataset/fetch_images/muri_images/*' + args.data_extension):
        image = cv2.imread(imageFileName)
        saliency = backprojection_saliency(image)
        #saliency = compute_sailency_through_backprop_color_spesific(image,0)
        #cv2.imshow("Image Saliency Threshold", saliency)
        #cv2.waitKey(0)
        cv2.imwrite(imageFileName+'salinecy.png',saliency*255)
        sailencyMaps.append(saliency)
    return sailencyMaps

# compute the saliency detection using backpropagation and return the masked image
def compute_sailency_through_backprop(image):
    mask = backprojection_saliency(image)
    segmentation = image * mask[:, :, np.newaxis]
    # cv2.imshow("Image Saliency Threshold", segmentation)
    # cv2.waitKey(0)
    return segmentation
# compute the saliency detection using backpropagation and return the masked image
def compute_sailency_through_backprop_color_spesific(image,idx):
    # It converts the BGR color space of image to HSV color space
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # show the output image on the newly created window
    # cv2.imshow('output image', hsv)
    # cv2.waitKey(0)
    # convert to LAB
    #colorLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # colorHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #
    # for label in threshold_dict.keys():
    #     #mask = colorLAB
    #     mask = colorHSV
    #     #LAB = threshold_dict[label]
    #     HSV = threshold_dict_hsv[label]
    #
    #     # lowerBound = np.array([0, LAB["A"][0] + 128, LAB["B"][0] + 128])
    #     # upperBound = np.array([255, LAB["A"][1] + 128, LAB["B"][1] + 128])
    #     lowerBound = np.array([HSV["H"][0], HSV["S"][0], HSV["V"][0]])
    #     upperBound = np.array([HSV["H"][1], HSV["S"][1], HSV["V"][1]])
    #
    #     # Masking
    #     mask = cv2.inRange(mask, lowerBound, upperBound)
    #     kernel = np.ones((5, 5), np.uint8)
    #     mask = cv2.erode(mask, kernel)
    #     kernel = np.ones((9, 9), np.uint8)
    #     mask = cv2.dilate(mask, kernel)
    #     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    #     if(len(cnts) != 0):
    #         cnt = max(cnts, key=cv2.contourArea)
    #         out = np.zeros(mask.shape, np.uint8)
    #         cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    #         mask = cv2.bitwise_and(mask, out)
    #         cv2.imshow('result', mask)
    #         cv2.waitKey(0)

    # Threshold of blue in HSV space
    # lower_blue = np.array([60, 35, 140])
    # upper_blue = np.array([179, 255, 255])
    # lower_red = np.array([160,20,70])
    # upper_red = np.array([190,255,255])
    #
    # # preparing the mask to overlay
    # maskblue = cv2.inRange(hsv, lower_blue, upper_blue)
    # #maskred = cv2.inRange(hsv, lower_red, upper_red)
    # #img = cv2.bitwise_and(image, image, mask=maskred)
    # img = cv2.bitwise_and(image, image, mask=maskblue)
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions

    img,mask = color_mask(image, threshold_colors['green'])
    #refine_saliency_with_grabcut(image, img)
    masker = np.zeros(img.shape)
    masker[mask] = 255
    cv2.imshow('result', masker)
    cv2.waitKey(0)
    # mask = backprojection_saliency(img)
    # segmentation = img * mask[:, :, np.newaxis]
    # cv2.imshow("Image Saliency Threshold", segmentation)
    # cv2.waitKey(0)
    return img

threshold_colors = {
            'red': [320, 30],
            'yellow': [30, 75],
            'blue':[150, 270],
            'green':[75, 150]
        }
# from matlab image filter
def color_mask(image, rangeHue):
    # figure out this bounding box
    imgPIL = im.fromarray(np.uint8(image)).convert('RGB')
    imageBox = imgPIL.getbbox()
    imgPIL = imgPIL.crop(imageBox)
    img = np.array(imgPIL)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    opencv_H = (hsv[:,:,0] / 180)
    opencv_S = (hsv[:,:,1] / 255)
    opencv_V = (hsv[:,:,2] / 255)
    rangeHue = np.array(rangeHue)/360
    if rangeHue[0] > rangeHue[1]:
        # red hue
        mask = np.logical_and(opencv_H > rangeHue[0],opencv_H <= 1) + \
               np.logical_and(opencv_H < rangeHue[1],opencv_H >= 0)
    else:
        # regular case
        mask = np.logical_and(opencv_H > rangeHue[0],opencv_H < rangeHue[1])
    #  % Saturation is modified according to the mask
    opencv_S = mask*opencv_S
    # this works but it would be better to just do the cropping
   # mask = np.logical_and(mask, opencv_S > 0.3)
    # now convert back
    hsv[:, :, 0] = np.round(opencv_H*180)
    hsv[:, :, 1] = np.round(opencv_S*255)
    hsv[:, :, 2] = np.round(opencv_V*255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb, mask

# using the model from https://github.com/Joker316701882/Salient-Object-Detection
class Saliency_NN:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        #with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
        self.saver = tf.train.import_meta_graph('./dataset/salience_model/my-model.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./dataset/salience_model'))
        self.image_batch = tf.get_collection('image_batch')[0]
        self.pred_mattes = tf.get_collection('mask')[0]

    # run neural network saliency
    def compute_saliency_NN(self,image):
        #with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
        origin_shape = image.shape
        rgb = np.expand_dims(
        cv2.resize(image.astype(np.uint8), [320, 320], interpolation=cv2.INTER_NEAREST).astype(np.float32) - self.g_mean, 0)
        feed_dict = {self.image_batch: rgb}
        pred_alpha = self.sess.run(self.pred_mattes, feed_dict=feed_dict)
        final_alpha = cv2.resize(np.squeeze(pred_alpha), np.flip(origin_shape[0:2])) * 255
        (T, saliency) = cv2.threshold(final_alpha, 255 / 2, 255, cv2.THRESH_BINARY)
        return refine_saliency_with_grabcut(image,saliency.astype(np.uint8))

    # grabcut finds the contors and returns the masked image
    def refine_saliency_with_grabcut(self,img, saliency):
        rect = largest_contours_rect(saliency)
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        saliency[np.where(saliency > 0)] = cv2.GC_FGD
        mask = saliency
        cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return mask

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS DATASET SALIENCY DETECTION')
    # directory data location
    parser.add_argument('--data_dir', type=str)
    # output directory to put blur measurement
    parser.add_argument('--threshold_dir', type=str)
    # type of data / image extension
    parser.add_argument('--data_extension', type=str,default='.png')
    args = parser.parse_args()
    # compute saliency detection
    #saliencyMaps = compute_saliency_Itti(args)
    compute_saliency_backprop(args)
