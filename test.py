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
import sys
from train import *

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

h = config.TRAIN.height
w = config.TRAIN.width

ni = int(math.ceil(np.sqrt(batch_size)))

def blurmap_3classes(index):
    print("Blurmap Generation")

    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_origonalResultswithnewDatasetwith256image'
    tl.files.exists_or_mkdir(save_dir_sample)

    #Put the input path!
    sharp_path = './input'
    test_sharp_img_list = os.listdir(sharp_path)
    test_sharp_img_list.sort()

    flag=0
    i=0

    for image in test_sharp_img_list:
        if(i>=index and i<index+100):
            print(i)
            if (image.find('.jpg') & image.find('.png') & image.find('.JPG')& image.find('.PNG')) != -1:

                sharp = os.path.join(sharp_path, image)
                sharp_image = Image.open(sharp)
                sharp_image.load()

                sharp_image = np.asarray(sharp_image, dtype="float32")

                if(len(sharp_image.shape)<3):
                    sharp_image= np.expand_dims(np.asarray(sharp_image), 3)
                    sharp_image=np.concatenate([sharp_image, sharp_image, sharp_image],axis=2)

                if (sharp_image.shape[2] ==4):
                    print(sharp_image.shape)
                    sharp_image = np.expand_dims(np.asarray(sharp_image), 3)

                    print(sharp_image.shape)
                    sharp_image = np.concatenate((sharp_image[:,:,0],sharp_image[:,:,1],sharp_image[:,:,2]),axis=2)

                print(sharp_image.shape)

                image_h, image_w =sharp_image.shape[0:2]
                print(image_h, image_w)

                test_image = sharp_image[0: image_h-(image_h%16), 0: 0 + image_w-(image_w%16), :]/(255.)

                # Model
                patches_blurred = tf.compat.v1.placeholder('float32', [1, test_image.shape[0], test_image.shape[1], 3], name='input_patches')
                if flag==0:
                    reuse =False
                else:
                    reuse =True

                start_time = time.time()

                with tf.compat.v1.variable_scope('Unified') as scope:
                    with tf.compat.v1.variable_scope('VGG') as scope3:
                        n, f0, f0_1, f1_2, f2_3, hrg, wrg = VGG19_pretrained(patches_blurred, reuse=reuse,scope=scope3)
                        #tl.visualize.draw_weights(n.all_params[0].eval(), second=10, saveable=True, name='weight_of_1st_layer', fig_idx=2012)

                    with tf.compat.v1.variable_scope('UNet') as scope1:
                        output,m1,m2,m3= Decoder_Network_classification(n, f0, f0_1, f1_2, f2_3 ,hrg,wrg, reuse = reuse, scope = scope1)

                    output_map = tf.nn.softmax(output.outputs)
                    output_map1 = tf.nn.softmax(m1.outputs)
                    output_map2 = tf.nn.softmax(m2.outputs)
                    output_map3 = tf.nn.softmax(m3.outputs)

                #a_vars = tl.layers.get_variables_with_name('Unified', False, True)

                saver = tf.compat.v1.train.Saver()
                sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False))
                tl.layers.initialize_global_variables(sess)

                # Load checkpoint
                saver.restore(sess, "./model/final_model.ckpt")

                start_time = time.time()
                blur_map,_,_,_ = sess.run([output_map,output_map1,output_map2,output_map3],{patches_blurred: np.expand_dims(
                    (test_image), axis=0)})
                blur_map = np.squeeze(blur_map)
                # o1= np.squeeze(o1)
                # o2 = np.squeeze(o2)
                # o3 = np.squeeze(o3)

                if ".jpg" in image:
                    image.replace(".jpg", ".png")
                    cv2.imwrite(save_dir_sample +  '/'+ image.replace(".jpg", ".png"), blur_map*255)
                if ".JPG" in image:
                    image.replace(".JPG", ".png")
                    cv2.imwrite(save_dir_sample +  '/'+ image.replace(".JPG", ".png"), blur_map*255)
                if ".PNG" in image:
                    image.replace(".jpg", ".png")
                    cv2.imwrite(save_dir_sample +  '/'+ image.replace(".jpg", ".png"), blur_map*255)
                if ".png" in image:
                    image.replace(".jpg", ".png")
                    cv2.imwrite(save_dir_sample + '/' + image.replace(".jpg", ".png"), blur_map*255)

                sess.close()
                flag=1

                print("5.--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                if(i==index+101-1):
                    return 0
        i = i + 1
    return 0


