#! /usr/bin/python
# -*- coding: utf8 -*-
import copy

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
from tensorlayer.models import Model

# model definition for neural network
# needs to be the same as the kim et al for loading the weights with the same names

# kim et al encoder
def VGG19_pretrained(t_image,reuse = False,scope="VGG"):
    with tf.compat.v1.variable_scope(scope, reuse=reuse) as vs:
        # input layer
        net_in = Input(t_image.shape, name='input') #tf.keras.layers.InputLayer(bgr, name='input')
        # conv1
        network = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                          name='conv1_1')(net_in)
        network = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv1_2')(network)
        f0 = network#Model(inputs=net_in,outputs=network)#,name='f0'
        #n.outputs= tf.nn.relu(n.outputs)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')(network)

        # conv2
        network = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv2_1')(network)
        network = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv2_2')(network)
        f0_1 = network#Model(inputs=net_in,outputs=network)#,name='f0_1'
        #n.outputs = tf.nn.relu(n.outputs)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')(network)

        # conv3
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_1')(network)
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_2')(network)
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_3')(network)
        network = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_4')(network)
        f1_2 = network#Model(inputs=net_in,outputs=network)#,name='f1_2'
        #n.outputs = tf.nn.relu(n.outputs)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')(network)
        # conv4
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_1')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_2')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_3')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv4_4')(network)
        f2_3 = network#Model(inputs=net_in,outputs=network)#,name='f2_3'
        #n.outputs=tf.nn.relu(n.outputs)
        network = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')(network)
        # conv5
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_1')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_2')(network)
        network = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv5_3')(network)
        n = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                   name='conv5_4')(network)
        n = Model(inputs=net_in,outputs=n)#,name=sope=c

    return net_in, n, f0, f0_1,f1_2,f2_3

# updated decoder
def Decoder_Network_classification(maininput, ninput,f0,f1_2,f2_3,f3_4, reuse=False, scope = "UNet"):
    # xavier_initializer was discontinued
    w_init1 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    w_init2 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    w_init3 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    w_init4 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    with tf.compat.v1.variable_scope(scope, reuse=reuse) as vs:
        #this bug..... w_init3->w_init4 # out_size=(hrg // 8, wrg // 8)
        # input layer
        n = DeConv2d(n_filter=512, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init4,
                     name='u4/d')(ninput.outputs)
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f3_4 = Conv2d(n_filter=512, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init4,
                      name='f3_4/c1')(f3_4)
        n = Elementwise(combine_fn=tf.add, act=tf.nn.relu, name='s5')([n, f3_4])
        #f3_4 = Model(inputs=n,outputs=f3_4,name='f3_4')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')

        # n = tf.nn.relu(n)# n.outputs
        # this is really studpid the origional code had the  name 'u34/c1' which was a typo but when i fixed it it cause
        # disscepencies with names. now running test 1 you have to name this u4/c1 but running test 2 you need the name u34/c1.
        n = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4,
                   name='u34/c1')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4,
                   name='u4/c2')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4,
                   name='u4/c3')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        # n.outputs = tf.nn.relu(n.outputs)
        m3BModel = Model(inputs=maininput,outputs=n)
        n_m3 = Conv2d(n_filter=5, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init4,
                      name='u4/loss3')(n)
        #(hrg // 4, wrg // 4)
        n = DeConv2d(n_filter=256, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init3,
                     name='u3/d')(n)
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        f2_3 = Conv2d(n_filter=256, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init3,
                      name='f2_3/c1')(f2_3)
        n = Elementwise(combine_fn=tf.add, act=tf.nn.relu, name='s4')([n, f2_3])
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')
        #n.outputs = tf.nn.relu(n)
        #n = tf.nn.relu(n)
        n = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3,
                   name='u3/c1')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3,
                   name='u3/c2')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3,
                   name='u3/c3')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        # n.outputs = tf.nn.relu(n.outputs)
        m2BModel = Model(inputs=maininput, outputs=n)
        n_m2 = Conv2d(n_filter=5, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init3,
                      name='u3/loss2')(n)
        # output-size= (hrg // 2, wrg // 2),
        n = DeConv2d(n_filter=128, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init2,
                     name='u2/d')(n)
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u2/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f1_2 = Conv2d(n_filter=128, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init2,
                      name='f1_2/c1')(f1_2)
        n = Elementwise(combine_fn=tf.add, act=tf.nn.relu, name='s3')([n, f1_2])
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu3')
        #n.outputs = tf.nn.relu(n)
        #n = tf.nn.relu(n)
        n = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init2,
                   name='u2/c1')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init2,
                   name='u2/c2')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        m1BModel = Model(inputs=maininput, outputs=n)
        n_m1 = Conv2d(n_filter=5, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init2,
                      name='u2/loss1')(n)
        #(hrg, wrg),
        n = DeConv2d(n_filter=64, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init1,
                     name='u1/d')(n)
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f0 = Conv2d(n_filter=64, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init1,
                    name='f0/c1')(f0)
        n = Elementwise(combine_fn=tf.add, act=tf.nn.relu, name='s2')([n, f0])
        # n.outputs = tf.nn.relu(n)
        #n = tf.nn.relu(n)
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu2')
        n = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1,
                   name='u1/c1')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1,
                   name='u1/c2')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        #n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1, name='u1/c3')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        # network = Conv2d(n_filter=3, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init1,
        #            name='u1/c5')(n)
        m4BModel = Model(inputs=maininput, outputs=n)

        network = Conv2d(n_filter=5, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init1,
                         name='u1/c5')(n)

        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        # n = ElementwiseLayer([n, n_init], tf.add, name='s1')
        #n.outputs = tf.nn.sigmoid(n.outputs)  # -> this is bug??

        # n2 = InputLayer(tf.nn.sigmoid(n.outputs), name='sigmoid')
        network = Model(inputs=maininput,outputs=network)#,name=scope
        n_m3 = Model(inputs=network.inputs,outputs=n_m3,)
        n_m2 = Model(inputs=network.inputs, outputs=n_m2,)
        n_m1 = Model(inputs=network.inputs, outputs=n_m1,)

    return network, n_m1, n_m2, n_m3,m1BModel,m2BModel,m3BModel,m4BModel

# kim et al version used for testing and getting weights
def Origional_Decoder_Network_classification(maininput, ninput,f0,f1_2,f2_3,f3_4, reuse=False, scope = "UNet"):
    # xavier_initializer was discontinued
    w_init1 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    w_init2 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    w_init3 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    w_init4 = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    with tf.compat.v1.variable_scope(scope, reuse=reuse) as vs:
        #this bug..... w_init3->w_init4 # out_size=(hrg // 8, wrg // 8)
        # input layer
        n = DeConv2d(n_filter=512, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init4,
                     name='u4/d')(ninput.outputs)
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f3_4 = Conv2d(n_filter=512, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init4,
                      name='f3_4/c1')(f3_4)
        n = Elementwise(combine_fn=tf.add, act=tf.nn.relu, name='s5')([n, f3_4])
        #f3_4 = Model(inputs=n,outputs=f3_4,name='f3_4')
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')

        # n = tf.nn.relu(n)# n.outputs
        # this is really studpid the origional code had the  name 'u34/c1' which was a typo but when i fixed it it cause
        # disscepencies with names. now running test 1 you have to name this u4/c1 but running test 2 you need the name u34/c1.
        n = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4,
                   name='u34/c1')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4,
                   name='u4/c2')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init4,
                   name='u4/c3')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m3 = Conv2d(n_filter=3, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init4,
                      name='u4/loss3')(n)
        #(hrg // 4, wrg // 4)
        n = DeConv2d(n_filter=256, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init3,
                     name='u3/d')(n)
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u3/b')
        f2_3 = Conv2d(n_filter=256, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init3,
                      name='f2_3/c1')(f2_3)
        n = Elementwise(combine_fn=tf.add, act=tf.nn.relu, name='s4')([n, f2_3])
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu4')
        #n.outputs = tf.nn.relu(n)
        #n = tf.nn.relu(n)
        n = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3,
                   name='u3/c1')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3,
                   name='u3/c2')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init3,
                   name='u3/c3')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u3/b3')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m2 = Conv2d(n_filter=3, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init3,
                      name='u3/loss2')(n)
        # output-size= (hrg // 2, wrg // 2),
        n = DeConv2d(n_filter=128, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init2,
                     name='u2/d')(n)
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u2/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f1_2 = Conv2d(n_filter=128, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init2,
                      name='f1_2/c1')(f1_2)
        n = Elementwise(combine_fn=tf.add, act=tf.nn.relu, name='s3')([n, f1_2])
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu3')
        #n.outputs = tf.nn.relu(n)
        #n = tf.nn.relu(n)
        n = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init2,
                   name='u2/c1')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init2,
                   name='u2/c2')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u2/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        n_m1 = Conv2d(n_filter=3, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init2,
                      name='u2/loss1')(n)
        #(hrg, wrg),
        n = DeConv2d(n_filter=64, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init1,
                     name='u1/d')(n)
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b')
        # n.outputs = tf.nn.relu(n.outputs)
        f0 = Conv2d(n_filter=64, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init1,
                    name='f0/c1')(f0)
        n = Elementwise(combine_fn=tf.add, act=tf.nn.relu, name='s2')([n, f0])
        # n.outputs = tf.nn.relu(n)
        #n = tf.nn.relu(n)
        # n = InputLayer(tf.nn.relu(n.outputs), name='relu2')
        n = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1,
                   name='u1/c1')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        n = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1,
                   name='u1/c2')(n)
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        #n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init1, name='u1/c3')
        # n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        # n.outputs = tf.nn.relu(n.outputs)
        # network = Conv2d(n_filter=3, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init1,
        #            name='u1/c5')(n)

        network = Conv2d(n_filter=3, filter_size=(1, 1), strides=(1, 1), act=None, padding='SAME', W_init=w_init1,
                         name='u1/c5')(n)

        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b2')
        # n.outputs = tf.nn.relu(n.outputs)
        # n = ElementwiseLayer([n, n_init], tf.add, name='s1')
        #n.outputs = tf.nn.sigmoid(n.outputs)  # -> this is bug??

        # n2 = InputLayer(tf.nn.sigmoid(n.outputs), name='sigmoid')
        network = Model(inputs=maininput,outputs=network)#,name=scope
        n_m3 = Model(inputs=network.inputs,outputs=n_m3,)
        n_m2 = Model(inputs=network.inputs, outputs=n_m2,)
        n_m1 = Model(inputs=network.inputs, outputs=n_m1,)

    return network, n_m1, n_m2, n_m3





