# coding=utf-8
import copy

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorlayer as tl
import numpy as np
import math
from config import config, log_config
from utils import *
from model import *
import matplotlib
import datetime
import time
import cv2
import argparse
import os
tf.compat.v1.disable_eager_execution()

def get_weights(sess,network):
    # https://github.com/TreB1eN/InsightFace_Pytorch/issues/137
    dict_weights_trained = np.load('/home/mary/code/NN_Blur_Detection_apc/inital_final_model.npy',allow_pickle=True)[()]
    params = []
    keys = dict_weights_trained.keys()
    for weights in network.trainable_weights:
        name = weights.name
        splitName ='/'.join(name.split('/')[2:-1])
        count_layers = 0
        for key in keys:
            keySplit = '/'.join(key.split(',')[1:])
            if '/d' in keySplit:
                keySplit = '/'.join(keySplit.split('/')[:-2])
            if splitName == keySplit:
                if 'bias' in name.split('/')[-1]:
                    params.append(dict_weights_trained[key][count_layers+1][0])
                else:
                    params.append(dict_weights_trained[key][count_layers][0])
                break
            count_layers = count_layers + 2

    sess.run(tl.files.assign_weights(params, network))

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

def load_and_save_npy_weights():
    # Model
    patches_blurred = tf.compat.v1.placeholder('float32', [1, 416, 640, 3], name='input_patches')
    reuse =False
    with tf.compat.v1.variable_scope('Unified') as scope:
        with tf.compat.v1.variable_scope('VGG') as scope3:
            input, n, f0, f0_1, f1_2, f2_3= VGG19_pretrained(patches_blurred, reuse=reuse,scope=scope3)
                        #tl.visualize.draw_weights(n.all_params[0].eval(), second=10, saveable=True, name='weight_of_1st_layer', fig_idx=2012)
        with tf.compat.v1.variable_scope('UNet') as scope1:
            output,m1,m2,m3= Decoder_Network_classification(input, n.outputs, f0.outputs, f0_1.outputs, f1_2.outputs,
                                                            f2_3.outputs,reuse = reuse,scope = scope1)

    a_vars = tl.layers.get_variables_with_name('Unified', True, True)

    #saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    get_weights(sess,output)
    print("loaded all the weights")

    # save checkpoint
    tl.files.save_ckpt(sess=sess, mode_name='final_checkpoint_tf2.ckpt', var_list=a_vars, printable=False)

if __name__ == '__main__':
    load_and_save_npy_weights()
