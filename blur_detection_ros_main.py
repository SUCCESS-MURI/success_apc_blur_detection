#!/usr/bin/env python3
import argparse
import rospy
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.training import py_checkpoint_reader
from model import Decoder_Network_classification, VGG19_pretrained
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sensor_msgs.msg import Image

def get_weights_checkpoint(sess,network,dict_weights_trained):
    # https://github.com/TreB1eN/InsightFace_Pytorch/issues/137
    #dict_weights_trained = np.load('./setup/final_model.npy',allow_pickle=True)[()]
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

    def __init__(self,model_checkpoint, height, width):
        self.model_checkpoint = model_checkpoint
        self.vgg_mean = np.array(([103.939, 116.779, 123.68])).reshape([1,1,3])
        # initialize the model and its weights
        self.setup_model(height, width)
        # setup the callback for the blur detection model
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/head_camera/rgb/image_raw",Image,self.callback)
        # if needed https://docs.ros.org/en/melodic/api/rospy/html/rospy.numpy_msg-module.html

        rospy.loginfo("Done Setting up Blur Detection")

    def callback(self,data):
        # convert to RGB image
        try:
            image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)
        # now run through model
        rgb = np.expand_dims(image - self.vgg_mean,axis = 0)
        blurMap = np.squeeze(self.sess.run([self.net_outputs], {self.net_regression.inputs: rgb}))
        blur_map = np.zeros((256,256))
        # numpy array with labels 
        blur_map[np.sum(blurMap[:,:] >= .2,axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:,:] >= .2,axis=2) == 1],
                                                                     axis=1)
        # uncertainty labeling
        blur_map[np.sum(blurMap[:, :] >= .2, axis=2) != 1] = 5
        # publish softmax output and image labeling
        # might need to create a message need to talk about this with alvika
        # https://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Creating_a_msg

    def setup_model(self, h, w):
        ### DEFINE MODEL ###
        patches_blurred = tf.compat.v1.placeholder('float32', [1, h, w, 3], name='input_patches')
        with tf.compat.v1.variable_scope('Unified'):
            with tf.compat.v1.variable_scope('VGG') as scope1:
                input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
            with tf.compat.v1.variable_scope('UNet') as scope2:
                self.net_regression, _, _, _, _, _, _, _ = Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3,
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
