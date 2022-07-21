#!/usr/bin/env python3
import rospy
import tensorflow as tf
# from skimage import transform
from skimage.io import imsave

tf.compat.v1.disable_eager_execution()
import tensorlayer as tl
from tensorflow.python.training import py_checkpoint_reader
from model import Decoder_Network_classification, VGG19_pretrained
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from std_msgs.msg import Int32
from success_apc_blur_detection.srv import BlurOutput, BlurOutputRequest, BlurOutputResponse
# import time
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import sys
# sys.path.insert(0, '/home/mary/code')
# from U2Net.model import U2NET
# from skimage import io, color
#from sensor_msgs.msg import Image
# from PIL import Image
# from torch.autograd import Variable
import pickle
import os
import torch

# from U2net
# normalize the predicted SOD probability map
# def normPRED(d):
#     ma = torch.max(d)
#     mi = torch.min(d)
#
#     dn = (d - mi) / (ma - mi)
#
#     return dn

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

    def __init__(self):
        self.model_checkpoint = rospy.get_param('~model_checkpoint')

        # self.unet_location = rospy.get_param('~unet_location')
        # self.define_and_get_unet()

        self.vgg_mean = np.array(([103.939, 116.779, 123.68])).reshape([1, 1, 3])
        # initialize the model and its weights
        self.setup_model(int(rospy.get_param('~height')), int(rospy.get_param('~width')))
        # self.define_and_get_unet(int(rospy.get_param('~unet_location')))
        # setup the callback for the blur detection model
        self.bridge = CvBridge()
        self.blur_srv = rospy.Service('~blurdetection', BlurOutput, self.blurdetection_callback)
        self.count = 0
        rospy.loginfo("Done Setting up Blur Detection")

    # def define_and_get_unet(self):
    #     model_dir = os.path.join(self.unet_location, 'U2Net', 'saved_models', 'u2net', 'u2net.pth')
    #     self.net = U2NET(3, 1)
    #     if torch.cuda.is_available():
    #         self.net.load_state_dict(torch.load(model_dir))
    #         self.net.cuda()
    #     else:
    #         self.net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    #     self.net.eval()

    def blurdetection_callback(self, req):
        try:
            # convert to RGB image
            try:
                image = self.bridge.imgmsg_to_cv2(req.bgr, "8UC3")
                # print(masked_image.shape)
            except CvBridgeError as e:
                print(e)
            # now run through saliancy model
            # test_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image_g = transform.resize(test_image, (320, 320), mode='constant')
            # tmpImg = np.zeros((image_g.shape[0], image_g.shape[1], 3))
            # image_g = image_g / np.max(image_g)
            # tmpImg[:, :, 0] = (image_g[:, :, 0] - 0.485) / 0.229
            # tmpImg[:, :, 1] = (image_g[:, :, 1] - 0.456) / 0.224
            # tmpImg[:, :, 2] = (image_g[:, :, 2] - 0.406) / 0.225
            # tmpImg = tmpImg.transpose((2, 0, 1))
            # inputs_test = torch.from_numpy(tmpImg).type(torch.FloatTensor)
            #
            # if torch.cuda.is_available():
            #     inputs_test = Variable(inputs_test.cuda().unsqueeze(0))
            # else:
            #     inputs_test = Variable(inputs_test.unsqueeze(0))
            #
            # d1, _, _, _, _, _, _ = self.net(inputs_test)
            #
            # # normalization
            # pred = d1[:, 0, :, :]
            # pred = normPRED(pred)
            #
            # predict = pred.squeeze().cpu().data.numpy()
            # im = Image.fromarray(predict * 255 > 1).convert('RGB')  # 1e-3
            # mask_main = np.array(im.resize((test_image.shape[1], test_image.shape[0]), resample=Image.BILINEAR))
            # mask_main = (mask_main[:, :, 0]).astype(np.uint8)

            rgb = np.expand_dims(image * 1. - self.vgg_mean, axis=0)
            blurMap = np.squeeze(self.sess.run([self.net_outputs], {self.net_regression.inputs: rgb}))
            # blur_map = np.zeros((blurMap.shape[0],blurMap.shape[1]))
            # numpy array with labels
            blur_map = np.argmax(blurMap, axis=2)

            # masked with salience image
            # blur_map_labels = np.squeeze(blur_map)
            # blur_map_labels[mask_main < 1e-3] = 10
            # blurmap_flap = blur_map_labels.flatten()
            # blurmap_flap = blurmap_flap[blurmap_flap != 10]

            # run through threshold and determine blur type
            num_0 = np.sum(blur_map == 0)
            num_1 = np.sum(blur_map == 1)
            num_2 = np.sum(blur_map == 2)
            num_3 = np.sum(blur_map == 3)
            num_4 = np.sum(blur_map == 4)
            label = np.argmax([num_0, num_1, num_2, num_3, num_4])

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
            # self.plot_images(image,rgb_blur_map)
            imsave(req.save_name, rgb_blur_map)
            msg_label = Int32()
            msg_label.data = label
            # publish softmax output and image labeling
            return BlurOutputResponse(success=True, label=msg_label)

        except Exception as e:
            rospy.logerr("Error - " + str(e))

        return BlurOutputResponse(success=False, msg=str(e))

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
                                                                                f2_3, reuse=False, scope=scope2)
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
