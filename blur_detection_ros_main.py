#!/usr/bin/env python3
import argparse
import rospy
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import tensorlayer as tl
from tensorflow.python.training import py_checkpoint_reader
from model import Decoder_Network_classification, VGG19_pretrained
from cv_bridge import CvBridge, CvBridgeError
from success_apc_blur_detection.msg import BlurDetectionOutput
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
#from sensor_msgs.msg import Image
#import time


def get_weights_checkpoint(sess, network, dict_weights_trained):
    # https://github.com/TreB1eN/InsightFace_Pytorch/issues/137
    params = []
    keys = dict_weights_trained.keys()
    for weights in network.trainable_weights:
        name = weights.name
        splitName = '/'.join(name.split(':')[:1])
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

    def __init__(self, args):
        self.model_checkpoint = args.model_checkpoint
        self.vgg_mean = np.array(([103.939, 116.779, 123.68])).reshape([1, 1, 3])
        # initialize the model and its weights
        self.setup_model(args.height, args.width)
        # setup the callback for the blur detection model
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/head_camera/rgb/image_raw",Image,self.callback)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        # if needed https://docs.ros.org/en/melodic/api/rospy/html/rospy.numpy_msg-module.html
        self.blur_detection_input_pub = rospy.Publisher('/blur_detection_output/input_image', CompressedImage,
                                                        queue_size=1)
        self.blur_detection_output_pub = rospy.Publisher('/blur_detection_output/output_image', CompressedImage,
                                                         queue_size=1)
        self.blur_detection_softmax_pub = rospy.Publisher('/blur_detection_output/softmax_output', CompressedImage,
                                                          queue_size=1)
        self.blur_detection_rgb_pub = rospy.Publisher('/blur_detection_output/rgb_output_image', CompressedImage,
                                                      queue_size=1)
        # self.count = 0
        rospy.loginfo("Done Setting up Blur Detection")
        # cv2.COLOR_BGR2RGB

    def callback(self, data):
        # convert to RGB image
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # now run through model
        # rospy.loginfo("I came here")
        # self.epoch_time = time.time()
        imageresized = cv2.resize(image, (self.height, self.width))
        rgb = np.expand_dims(imageresized * 1.0 - self.vgg_mean, axis=0)
        blurMap = np.squeeze(self.sess.run([self.net_outputs], {self.net_regression.inputs: rgb}))
        blur_map = np.zeros((blurMap.shape[0], blurMap.shape[1]))
        # blur_map = np.argmax(blurMap,axis=2)[:,:,np.newaxis].astype(np.int32)
        # numpy array with labels 
        # .4 was found from validation data results with different uncertainty levels tested
        blur_map[np.sum(blurMap[:, :] >= .4, axis=2) == 1] = np.argmax(
            blurMap[np.sum(blurMap[:, :] >= .4, axis=2) == 1],
            axis=1)
        # uncertainty labeling
        blur_map[np.sum(blurMap[:, :] >= .4, axis=2) != 1] = 5
        blur_map = blur_map.astype(np.uint8)
        # now make rgb image 
        # now color code
        rgb_blur_map = np.zeros((blur_map.shape[0], blur_map.shape[1], 3))
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
        # publish softmax output and image labeling
        # https://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Creating_a_msg
        self.publish_BlurDetectionOutput(image, blur_map, blurMap, rgb_blur_map)

    # http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
    def publish_BlurDetectionOutput(self, input_image, output_image, softmax_image_output, rgb_output):
        time = rospy.Time.now()
        msginput = CompressedImage()
        msginput.header.stamp = time
        msginput.format = "jpeg"
        msginput.data = np.array(cv2.imencode('.jpg', input_image)[1]).tostring()
        msgoutput = CompressedImage()
        msgoutput.header.stamp = time
        msgoutput.format = "jpeg"
        msgoutput.data = np.array(cv2.imencode('.jpg', output_image)[1]).tostring()
        msgsoftmax = CompressedImage()
        msgsoftmax.header.stamp = time
        msgsoftmax.format = "jpeg"
        msgsoftmax.data = np.array(cv2.imencode('.jpg', softmax_image_output)[1]).tostring()
        msgsrgb = CompressedImage()
        msgsrgb.header.stamp = time
        msgsrgb.format = "jpeg"
        msgsrgb.data = np.array(cv2.imencode('.jpg', rgb_output)[1]).tostring()
        self.blur_detection_input_pub.publish(msginput)
        self.blur_detection_output_pub.publish(msgoutput)
        self.blur_detection_softmax_pub.publish(msgsoftmax)
        self.blur_detection_rgb_pub.publish(msgsrgb)
        # msg = self.bridge.cv2_to_imgmsg(input_image, encoding="bgr8")
        # msg.header.stamp = time
        # self.blur_detection_input_pub.publish(msg)
        # msg = self.bridge.cv2_to_imgmsg(output_image, encoding="passthrough")
        # msg.header.stamp = time
        # self.blur_detection_output_pub.publish(msg)
        # msg = self.bridge.cv2_to_imgmsg(softmax_image_output, encoding="passthrough")
        # msg.header.stamp = time
        # self.blur_detection_softmax_pub.publish(msg)
        # msg = self.bridge.cv2_to_imgmsg(rgb_output, encoding="bgr8")
        # msg.header.stamp = time
        # self.blur_detection_rgb_pub.publish(msg)

    def setup_model(self, h, w):
        self.height = h
        self.width = w
        ### DEFINE MODEL ###
        patches_blurred = tf.compat.v1.placeholder('float32', [1, self.height, self.width, 3], name='input_patches')
        with tf.compat.v1.variable_scope('Unified'):
            with tf.compat.v1.variable_scope('VGG') as scope1:
                input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
            with tf.compat.v1.variable_scope('UNet') as scope2:
                self.net_regression, _, _, _, _, _, _, _ = Decoder_Network_classification(input, n, f0, f0_1, f1_2,
                                                                                          f2_3,
                                                                                          reuse=False, scope=scope2)
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
    parser.add_argument('--model_checkpoint', type=str, help='model checkpoint name and location for using',
                        default='/home/mary/catkin_ws/src/success_apc_blur_detection/model/Final_MURI_Model.ckpt')
    # parser.add_argument('--height', type=int,default=256)
    # parser.add_argument('--width', type=int,default=256)
    # CMU
    # parser.add_argument('--height', type=int,default=480)
    # parser.add_argument('--width', type=int,default=640)
    # for speeding up the runs
    width = int(640 * 50 / 100)
    height = int(480 * 50 / 100)
    parser.add_argument('--height', type=int, default=height)
    parser.add_argument('--width', type=int, default=width)
    # BYU
    # parser.add_argument('--height', type=int,default=720)
    # parser.add_argument('--width', type=int,default=1280)
    args = parser.parse_args()

    tl.global_flag['model_checkpoint'] = args.model_checkpoint
    blur_detection = BlurDetection(args)
    # Make sure sim time is working
    while not rospy.Time.now():
        pass
    # spin
    rospy.spin()
