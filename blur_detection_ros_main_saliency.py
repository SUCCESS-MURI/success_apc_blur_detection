#!/usr/bin/env python3
import argparse
import os

import rospy
import tensorflow as tf
import torch
from skimage import transform
from torch.autograd import Variable
from skimage.io import imsave

tf.compat.v1.disable_eager_execution()
import tensorlayer as tl
from tensorflow.python.training import py_checkpoint_reader
from model import Updated_Decoder_Network_classification, VGG19_pretrained
from cv_bridge import CvBridge, CvBridgeError
# from success_apc_blur_detection.msg import BlurDetectionOutput
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from success_apc_blur_detection.srv import BlurMaskOutput, BlurMaskOutputRequest, BlurMaskOutputResponse
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_weights_checkpoint(sess,network,dict_weights_trained):
    # https://github.com/TreB1eN/InsightFace_Pytorch/issues/137
    params = []
    keys = dict_weights_trained.keys()
    for weights in network.trainable_weights:
        name = weights.name
        splitName ='/'.join(name.split(':')[:1])
        for key in keys:
            keySplit = '/'.join(key.split('/'))
            if splitName == keySplit:
                if 'bias' in name.split('/')[-1]:
                    params.append(dict_weights_trained[key])
                else:
                    params.append(dict_weights_trained[key])
                break

    sess.run(tl.files.assign_weights(params, network))

# class for executing the blur detection algorithm
class BlurDetection:

    def __init__(self):
        self.model_checkpoint = rospy.get_param('~model_checkpoint')
        self.vgg_mean = np.array(([103.939, 116.779, 123.68])).reshape([1,1,3])
        # initialize the model and its weights
        self.setup_model(int(rospy.get_param('~height')), int(rospy.get_param('~width')))
        # self.define_and_get_unet(int(rospy.get_param('~unet_location')))
        # setup the callback for the blur detection model
        self.bridge = CvBridge()
        self.blur_srv = rospy.Service('~blurdetection', BlurMaskOutput, self.blurdetection_callback)
        self.count = 0
        rospy.loginfo("Done Setting up Blur Detection")

    def blurdetection_callback(self,req):
        try:
            # convert to RGB image
            try:
                image = self.bridge.imgmsg_to_cv2(req.bgr, "8UC3")
                # print(image.shape)
                masked_image = self.bridge.imgmsg_to_cv2(req.mask, "passthrough")
                # print(masked_image.shape)
            except CvBridgeError as e:
                print(e)
            # now run through model
            rgb = np.expand_dims(image*1. - self.vgg_mean, axis=0)
            blurMap = np.squeeze(self.sess.run([self.net_outputs], {self.net_regression.inputs: rgb}))
            # numpy array with labels
            blur_map = np.argmax(blurMap,axis=2)

            # masked with salience image
            blur_map_labels = np.squeeze(blur_map)
            blur_map_labels[masked_image < 1e-3] = 10
            blurmap_flap = blur_map_labels.flatten()
            blurmap_flap = blurmap_flap[blurmap_flap != 10]

            # run through threshold and determine blur type
            num_0 = np.sum(blurmap_flap == 0)
            num_1 = np.sum(blurmap_flap == 1)
            num_2 = np.sum(blurmap_flap == 2)
            num_3 = np.sum(blurmap_flap == 3)
            num_4 = np.sum(blurmap_flap == 4)
            label = np.argmax([num_0, num_1, num_2, num_3, num_4])

            blur_map = blur_map.astype(np.uint8)
            # now make rgb image
            # now color code
            rgb_blur_map = np.zeros((blur_map.shape[0],blur_map.shape[1],3))
            # blue motion blur
            rgb_blur_map[blur_map == 1] = [255, 0, 0]
            # green focus blur
            rgb_blur_map[blur_map == 2] = [0, 255, 0]
            # red darkness blur
            rgb_blur_map[blur_map == 3] = [0, 0, 255]
            # pink brightness blur
            rgb_blur_map[blur_map == 4] = [255, 192, 203]
            # yellow unknown blur
            rgb_blur_map[blur_map == 5] = [0, 255, 255]
            rgb_blur_map = rgb_blur_map.astype(np.uint8)
            #self.plot_images(image,rgb_blur_map)
            imsave(req.save_name, rgb_blur_map)
            msg_label = Int32()
            msg_label.data = label
            # publish softmax output and image labeling
            return BlurMaskOutputResponse(success=True, label=msg_label)

        except Exception as e:
            rospy.logerr("Error - " + str(e))

        return BlurMaskOutputResponse(success=False, msg=str(e))

    def setup_model(self, h, w):
        self.height = h
        self.width = w
        ### DEFINE MODEL ###
        patches_blurred = tf.compat.v1.placeholder('float32', [1, self.height, self.width, 3], name='input_patches')
        with tf.compat.v1.variable_scope('Unified'):
            with tf.compat.v1.variable_scope('VGG') as scope1:
                input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
            with tf.compat.v1.variable_scope('UNet') as scope2:
                self.net_regression, _, _, _ = Updated_Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3,
                                                                                     reuse=False,scope=scope2)
        # Load checkpoint
        # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
        configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        configTf.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configTf)
        tl.layers.initialize_global_variables(self.sess)
        # read check point weights
        reader = py_checkpoint_reader.NewCheckpointReader(self.model_checkpoint)
        state_dict = {v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()}
        # save weights to the model
        get_weights_checkpoint(self.sess, self.net_regression, state_dict)
        output_map = tf.nn.softmax(self.net_regression.outputs)
        self.net_outputs = output_map

if __name__ == "__main__":
    rospy.init_node("Blur_Detection")
    blur_detection = BlurDetection()
    # Make sure sim time is working
    while not rospy.Time.now():
        pass
    # spin
    rospy.spin()
