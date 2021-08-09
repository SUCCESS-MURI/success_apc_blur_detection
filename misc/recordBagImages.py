#!/usr/bin/env python3

import argparse
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from success_apc_blur_detection.msg import BlurDetectionOutput
import os
import numpy as np

class image_converter:
    
    def __init__(self):
        # if not os.path.exists(output_folder):
        #     os.mkdir(output_folder)
        # self.output_path = output_folder
        self.count = 0
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/blur_detection_output",BlurDetectionOutput,self.callback)
    
    def callback(self,data):
        # convert to RGB image
        try:
            image_input = self.bridge.imgmsg_to_cv2(data.input_image, "passthrough")
            image_w_uncertanity = self.bridge.imgmsg_to_cv2(data.image_with_uncertanity, "passthrough")
            softmax_image = self.bridge.imgmsg_to_cv2(data.softmax_output,"passthrough")
        except CvBridgeError as e:
            print(e)
        filename = "/home/mary/code/local_success_dataset/muri/muri_images_run/BYU_image_data/BYU_image_data_rgb_image_input-{}.png".format(self.count)
        cv2.imwrite(filename, image_input) 
        filename = "/home/mary/code/local_success_dataset/muri/muri_images_run/BYU_image_data/BYU_image_data_rgb_image_output-{}.png".format(self.count)
        cv2.imwrite(filename, image_w_uncertanity) 
        filename = "/home/mary/code/local_success_dataset/muri/muri_images_run/BYU_image_data/BYU_image_data_rgb_image_softmax_output-{}.npy".format(self.count)
        np.save(filename, softmax_image)
        self.count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--output_path', type=str,default='/home/mary/code/local_success_dataset/muri/muri_images_converted/block_perception_0.5_3')
    args = parser.parse_args()
    rospy.init_node('collect_images_from_bd')
    img = image_converter()
    rospy.spin()