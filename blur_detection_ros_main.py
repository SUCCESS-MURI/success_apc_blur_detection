#!/usr/bin/env python3
import argparse

import rospy
import tensorflow as tf
from model import *

class BlurDetection:

    def __init__(self,model_checkpoint):
        self.model_checkpoint = model_checkpoint
        # initialize the model and its weights
        self.setup_model()
        # connect to docking node.
        # rospy.loginfo("Connecting to %s..." % self.UNDOCK_ACTION_NAME)
        # self.client_undock = actionlib.SimpleActionClient(self.UNDOCK_ACTION_NAME, UndockAction)
        # self.client_undock.wait_for_server()
        # rospy.loginfo("Done.")
        # rospy.loginfo("Connecting to %s..." % self.DOCK_ACTION_NAME)
        # self.client_dock = actionlib.SimpleActionClient(self.DOCK_ACTION_NAME, DockAction)
        # self.client_dock.wait_for_server()
        # rospy.loginfo("Done.")

    def callback(self,data):
        # convert to RGB image
        # now run through model
        # return result
        pass

    def setup_model(self):
        ### DEFINE MODEL ###
        patches_blurred = tf.compat.v1.placeholder('float32', [1, 256, 256, 3], name='input_patches')
        #classification_map = tf.compat.v1.placeholder('int64', [1, 256, 256, 1], name='labels')
        with tf.compat.v1.variable_scope('Unified'):
            with tf.compat.v1.variable_scope('VGG') as scope1:
                input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
            with tf.compat.v1.variable_scope('UNet') as scope2:
                net_regression, _, _, _, _, _, _, _ = Decoder_Network_classification(input, n, f0, f0_1,
                                                                                                f1_2, f2_3, reuse=False,
                                                                                                scope=scope2)

        output_map = tf.nn.softmax(net_regression.outputs)#tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
        self.net_outputs = output_map
        # load the model checkpoint and weights

if __name__ == "__main__":
    rospy.init_node("Blur_Detection")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, help='model checkpoint name and location for using')
    args = parser.parse_args()

    tl.global_flag['model_checkpoint'] = args.model_checkpoint
    blur_detection = BlurDetection(args.model_checkpoint)
    # Make sure sim time is working
    while not rospy.Time.now():
        pass
    # spin
    rospy.spin()
