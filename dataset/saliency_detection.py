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

##########################################################################3

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
