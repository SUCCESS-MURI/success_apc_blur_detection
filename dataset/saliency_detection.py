# Author mhatfalv
import os

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

################# folliowing code from ###########
# https://github.com/jacobgil/saliency-from-backproj/blob/master/saliency.py
# written by Jacob Gildenblat
# jacobgil
# compute the contours
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

# using the model from https://github.com/Joker316701882/Salient-Object-Detection
class Saliency_NN:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        #with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
        self.saver = tf.train.import_meta_graph('./salience_model/my-model.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./salience_model'))
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
