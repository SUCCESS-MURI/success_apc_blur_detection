# coding=utf-8
import copy
import multiprocessing
import random

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorlayer as tl
import numpy as np
import math

from tensorflow.python.training import py_checkpoint_reader

from config import config, log_config
from setup.loadNPYWeightsSaveCkpt import get_weights
from utils import *
from model import *
import matplotlib
import datetime
import time
import cv2
import argparse
import os
#tf.compat.v1.disable_eager_execution()
#import sys

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

h = config.TRAIN.height
w = config.TRAIN.width

ni = int(math.ceil(np.sqrt(batch_size)))
# np.random.seed(10)
# random.seed(10)

def read_all_imgs(img_list, path='', n_threads=32, mode = 'RGB'):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        if mode == 'RGB':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGB_fn, path=path)
        elif mode == 'GRAY':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_GRAY_fn, path=path)
        elif mode == 'RGB2GRAY':
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_RGBGRAY_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

# https://izziswift.com/better-way-to-shuffle-two-numpy-arrays-in-unison/
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

# training with the original CHUK images
def train_with_CUHK():
    checkpoint_dir ="test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    save_dir_sample = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    input_path = config.TRAIN.CUHK_blur_path  #for comparison with neurocomputing
    train_blur_img_list = sorted(tl.files.load_file_list(path=input_path, regx='(out_of_focus|motion).*.(jpg|JPG)',
                                                         printable=False))
    train_mask_img_list=[]

    for str in train_blur_img_list:
        if ".jpg" in str:
            train_mask_img_list.append(str.replace(".jpg",".png"))
        else:
            train_mask_img_list.append(str.replace(".JPG", ".png"))

    gt_path = config.TRAIN.CUHK_gt_path
    print(train_blur_img_list)

    train_blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=batch_size ,mode='RGB')
    train_mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=batch_size,mode='GRAY')
    train_edge_imgs = []
    for img in train_blur_imgs:
        edges = cv2.Canny(img, 100, 200)
        train_edge_imgs.append(edges)

    index= 0
    train_classification_mask= []
    #img_n = 0
    for img in train_mask_imgs:
        if(index<236):
            tmp_class = img
            tmp_classification = np.concatenate((img,img,img),axis = 2)
            tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp
            tmp_class[np.where(tmp_classification[:,:,0]>0)] =1 #defocus blur
        else:
            tmp_class = img
            tmp_classification = np.concatenate((img, img, img), axis=2)
            tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp
            tmp_class[np.where(tmp_classification[:,:,0]>0)] =2 #defocus blur

        train_classification_mask.append(tmp_class)
        index =index +1

    ### DEFINE MODEL ###
    # gpu allocation
    # device_type = 'GPU'
    # devices = tf.config.experimental.list_physical_devices(
    #     device_type)
    # devices_names = [d.name.split("e:")[1]for d in devices]
    #strategy = tf.compat.v1.distribute.MirroredStrategy(devices=devices_names,
                                                      #  cross_device_ops=tf.distribute.ReductionToOneDevice())
    patches_blurred = tf.compat.v1.placeholder('float32', [batch_size, h, w, 3], name = 'input_patches')
    #labels_sigma = tf.compat.v1.placeholder('float32', [batch_size,h,w, 1], name = 'lables')
    classification_map= tf.compat.v1.placeholder('int32', [batch_size, h, w,1], name='labels')
    #with strategy.scope():
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred,reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression,m1,m2,m3= Decoder_Network_classification(input, n, f0, f0_1,f1_2, f2_3,reuse = False,
                                                                    scope = scope2)

        ### DEFINE LOSS ###
        loss1 = tl.cost.cross_entropy((net_regression.outputs),  tf.squeeze( classification_map), name='loss1')
        loss2 = tl.cost.cross_entropy((m1.outputs), tf.squeeze( tf.image.resize(classification_map, [int(h/2), int(w/2)],
                                                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )),name ='loss2')
        loss3 = tl.cost.cross_entropy((m2.outputs), tf.squeeze( tf.image.resize(classification_map, [int(h/4), int(w/4)],
                                                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) ),name='loss3')
        loss4 = tl.cost.cross_entropy((m3.outputs), tf.squeeze( tf.image.resize(classification_map, [int(h/8), int(w/8)],
                                                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )),name='loss4')
        out =(net_regression.outputs)
        loss = loss1 + loss2 + loss3 + loss4

    with tf.compat.v1.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init*0.1*0.1, trainable = False)
        lr_v2 = tf.Variable(lr_init*0.1, trainable=False)

    ### DEFINE OPTIMIZER ###
    a_vars = tl.layers.get_variables_with_name('Unified', False, True)  #
    var_list1 = tl.layers.get_variables_with_name('VGG', True, True)  # ?
    var_list2 = tl.layers.get_variables_with_name('UNet', True, True)  # ?
    opt1 = tf.optimizers.Adam(learning_rate=lr_v)# *0.1*0.1
    opt2 = tf.optimizers.Adam(learning_rate=lr_v2)
    grads = tf.gradients(ys=loss, xs=var_list1 + var_list2,unconnected_gradients='zero')
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op = tf.group(train_op1, train_op2)
    configTf = tf.compat.v1.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    configTf.gpu_options.allow_growth = True
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)

    #     except RuntimeError as e:
    #         print(e)
    sess = tf.compat.v1.Session(config=configTf)

    print("initializing global variable...")
    tl.layers.initialize_global_variables(sess)
    print("initializing global variable...DONE")

    ### initalize weights ###
    # initialize weights from previous run if indicated
    if tl.global_flag['start_from'] != 0:
        print("loading initial checkpoint")
        tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                           save_dir=checkpoint_dir, var_list=a_vars, is_latest=True,printable=True)
    else:
        ### LOAD VGG ###
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npy from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1',allow_pickle=True).item()
        #
        params = []
        count_layers = 0
        for val in sorted(npz.items()):
            if (count_layers < 16):
                W = np.asarray(val[1][0])
                b = np.asarray(val[1][1])
                print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
                params.extend([W, b])
            count_layers += 1

        sess.run(tl.files.assign_weights(params, n))

    ### START TRAINING ###
    augmentation_list = [0, 1]
    train_blur_imgs = np.array(train_blur_imgs, dtype=object)
    train_classification_mask = np.array(train_classification_mask, dtype=object)
    for epoch in range(tl.global_flag['start_from'], n_epoch + 1):
        # update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            #new_lr_decay = lr_decay ** (epoch // decay_every)
            #new_lr_decay = new_lr_decay * lr_decay
            sess.run(tf.compat.v1.assign(lr_v, lr_v * lr_decay))
            sess.run(tf.compat.v1.assign(lr_v2, lr_v2 * lr_decay))
            log = " ** new learning rate for Encoder: %f and for Decoder %f\n" % (sess.run(lr_v), sess.run(lr_v2))
            # print(log)
            with open(checkpoint_dir + "/training_CHUK_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
        elif epoch == tl.global_flag['start_from']:
            log = " ** init lr for Decoder: %f Encoder: %f decay_every_init: %d, lr_decay: %f\n" % (sess.run(lr_v),
                                                                                sess.run(lr_v2), decay_every, lr_decay)
            # print(log)
            with open(checkpoint_dir + "/training_CHUK_metrics.log", "a") as f:
                # perform file operations
                f.write(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0
        new_batch_size = batch_size  #batchsize 50->40 + 10(augmented)

        #data suffle***
        train_blur_imgs, train_classification_mask = unison_shuffled_copies(train_blur_imgs,train_classification_mask)

        for idx in range(0, len(train_blur_imgs), new_batch_size):
            step_time = time.time()

            # augmentation_list = [0, 1]
            augmentation = random.choice(augmentation_list)
            if augmentation == 0:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],
                                train_classification_mask[idx: idx + new_batch_size])],
                                fn=crop_sub_img_and_classification_fn)
            elif augmentation == 1:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],
                                                                            train_classification_mask[
                                                                            idx: idx + new_batch_size])],
                                                            fn=crop_sub_img_and_classification_fn_aug)
            # print images_and_score.shape
            imlist, clist = images_and_score.transpose((1, 0, 2, 3, 4))
            # print clist.shape
            clist = clist[:, :, :, 0]
            # print clist.shape
            clist = np.expand_dims(clist, axis=3)

            err,l1,l2,l3,l4, _, outmap = sess.run([loss,loss1,loss2,loss3,loss4,train_op,out], {net_regression.inputs: imlist,
                                                                                           classification_map: clist})

            # outmap1 = np.squeeze(outmap[1,:,:,0])
            # outmap2 = np.squeeze(outmap[1, :, :, 1])
            # outmap3 = np.squeeze(outmap[1, :, :, 2])

            #if(idx%100 ==0):
                # cv2.imwrite(save_dir_sample + '/input_mask.png', np.squeeze(clist[1, :, :, 0]))
                # cv2.imwrite(save_dir_sample + '/input.png', np.squeeze(imlist[1,:,:,:]))
                # cv2.imwrite(save_dir_sample + '/im.png', outmap1)
                # cv2.imwrite(save_dir_sample + '/im1.png', outmap2)
                # cv2.imwrite(save_dir_sample + '/im2.png', outmap3)
            # https://matthew-brett.github.io/teaching/string_formatting.html
            #print(
            #     "Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3: %.6f,loss4: %.6f" % (
            #     epoch, n_epoch, n_iter, time.time() - step_time, err, l1, l2, l3, l4))
            # metrics_file.write("Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3: %.6f,loss4:
            # %.6f" % (epoch, n_epoch, n_iter, time.time() - step_time, err,l1,l2,l3,l4))
            total_loss += err
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f\n" % (epoch, n_epoch, time.time() - epoch_time,
                                                                         total_loss/n_iter)
        #only way to write to log file while running
        with open(checkpoint_dir+"/training_CHUK_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        ## save model
        if epoch % 10 == 0:
            tl.files.save_ckpt(sess=sess,mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                            save_dir = checkpoint_dir, var_list = a_vars, global_step = epoch, printable = False)

# train with synthetic images
def train_with_synthetic():
    checkpoint_dir ="test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    save_dir_sample = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    input_path = config.TRAIN.synthetic_blur_path
    train_blur_img_list = sorted(tl.files.load_file_list(path=input_path, regx='(out_of_focus|motion).*.(jpg|JPG)',
                                                         printable=False))
    train_mask_img_list=[]

    for str in train_blur_img_list:
        if ".jpg" in str:
            train_mask_img_list.append(str.replace(".jpg",".png"))
        else:
            train_mask_img_list.append(str.replace(".JPG", ".png"))

    #augmented dataset read
    gt_path =config.TRAIN.synthetic_gt_path
    print(train_mask_img_list)

    train_blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=100, mode='RGB')
    train_mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=100, mode='GRAY')

    index= 0
    train_classification_mask= []
    #print train_mask_imgs
    #img_n = 0
    for img in train_mask_imgs:

        tmp_class = img
        tmp_classification = np.concatenate((img,img,img),axis = 2)

        tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp
        tmp_class[np.where(tmp_classification[:,:,0]==100)] =1 #motion blur
        tmp_class[np.where(tmp_classification[:,:,0]==200)] =2 #defocus blur

        train_classification_mask.append(tmp_class)
        index =index +1

    input_path2 = config.TRAIN.CUHK_blur_path
    ori_train_blur_img_list = sorted(tl.files.load_file_list(path=input_path2, regx='(out_of_focus|motion).*.(jpg|JPG)',
                                                             printable=False))
    ori_train_mask_img_list=[]

    for str in ori_train_blur_img_list:
        if ".jpg" in str:
            ori_train_mask_img_list.append(str.replace(".jpg",".png"))
        else:
            ori_train_mask_img_list.append(str.replace(".JPG", ".png"))

    #augmented dataset read
    gt_path2 = config.TRAIN.CUHK_gt_path
    print(train_blur_img_list)

    ori_train_blur_imgs = read_all_imgs(ori_train_blur_img_list, path=input_path2, n_threads=batch_size ,mode='RGB')
    ori_train_mask_imgs = read_all_imgs(ori_train_mask_img_list, path=gt_path2, n_threads=batch_size,mode='GRAY')
    train_edge_imgs = []

    index= 0
    ori_train_classification_mask= []
    #img_n = 0
    for img in ori_train_mask_imgs:
        if(index<236):
            tmp_class = img
            tmp_classification = np.concatenate((img,img,img),axis = 2)

            tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp
            tmp_class[np.where(tmp_classification[:,:,0]>0)] =1 #defocus blur
        else:
            tmp_class = img
            tmp_classification = np.concatenate((img, img, img), axis=2)
            tmp_class[np.where(tmp_classification[:,:,0]==0)] =0 #sharp
            tmp_class[np.where(tmp_classification[:,:,0]>0)] =2 #defocus blur

        ori_train_classification_mask.append(tmp_class)
        index =index +1
    train_mask_imgs=  train_classification_mask
    #for i in range(10):
    train_blur_imgs = train_blur_imgs + ori_train_blur_imgs
    train_mask_imgs = train_mask_imgs + ori_train_classification_mask

    print(len(train_blur_imgs), len(train_mask_imgs))
    ori_train_classification_mask = None
    tmp_class = None
    tmp_classification = None
    ori_train_blur_imgs = None
    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [batch_size, h, w, 3], name = 'input_patches')
    #labels_sigma = tf.compat.v1.placeholder('float32', [batch_size,h,w, 1], name = 'lables')
    classification_map= tf.compat.v1.placeholder('int32', [batch_size, h, w,1], name='labels')
    #class_map = tf.placeholder('int32', [batch_size, h, w], name='classes')
    #attention_edge = tf.placeholder('float32', [batch_size, h, w, 1], name='attention')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred,reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression,m1,m2,m3= Decoder_Network_classification(input, n.outputs, f0.outputs, f0_1.outputs, f1_2.outputs,
                                                                    f2_3.outputs,reuse = False,scope = scope2)
    ### DEFINE LOSS ###
    loss1 = tl.cost.cross_entropy((net_regression.outputs),  tf.squeeze( classification_map), name='loss1')
    loss2 = tl.cost.cross_entropy((m1.outputs),   tf.squeeze( tf.image.resize(classification_map, [int(h/2), int(w/2)],
                                                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )),name ='loss2')
    loss3 = tl.cost.cross_entropy((m2.outputs),   tf.squeeze( tf.image.resize(classification_map, [int(h/4), int(w/4)],
                                                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR) ),name='loss3')
    loss4 = tl.cost.cross_entropy((m3.outputs), tf.squeeze( tf.image.resize(classification_map, [int(h/8), int(w/8)],
                                                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR )),name='loss4')
    out =(net_regression.outputs)
    loss = loss1 + loss2 + loss3 + loss4

    with tf.compat.v1.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable = False)
    #
    # ### DEFINE OPTIMIZER ###
    a_vars = tl.layers.get_variables_with_name('Unified', False, True)  #
    var_list1 = tl.layers.get_variables_with_name('VGG', True, True)  # ?
    var_list2 = tl.layers.get_variables_with_name('UNet', True, True) #?t_vars
    opt1 = tf.compat.v1.train.AdamOptimizer(lr_v)#*0.1*0.1
    opt2 = tf.compat.v1.train.AdamOptimizer(lr_v*0.1)
    grads = tf.gradients(ys=loss, xs=var_list1 + var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op = tf.group(train_op1, train_op2)
    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #
    #     except RuntimeError as e:
    #         print(e)
    sess = tf.compat.v1.Session(config=configTf)
    print("initializing global variable...")
    tl.layers.initialize_global_variables(sess)
    print("initializing global variable...DONE")

    ### initial checkpoint ###
    if tl.global_flag['start_from'] != 0:
        tl.files.load_ckpt(sess=sess, mode_name='final_SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                           save_dir=checkpoint_dir,var_list=a_vars, is_latest=True, printable=True)
    else:
        tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                           save_dir=checkpoint_dir,var_list=a_vars,is_latest=True, printable=True)

    ### START TRAINING ###
    augmentation_list = [0, 1]
    sess.run(tf.compat.v1.assign(lr_v, lr_init))
    train_blur_imgs = np.array(train_blur_imgs,dtype=object)
    train_mask_imgs = np.array(train_mask_imgs,dtype=object)
    imagesmaskZipped = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs, train_mask_imgs)],
                                                fn=crop_sub_img_and_classification_fn)
    # print images_and_score.shape
    train_blur_imgs, train_mask_imgs = imagesmaskZipped.transpose((1, 0, 2, 3, 4))
    # print clist.shape
    train_mask_imgs = train_mask_imgs[:, :, :, 0]
    # print clist.shape
    train_mask_imgs = np.expand_dims(train_mask_imgs, axis=3)
    for epoch in range(tl.global_flag['start_from'], n_epoch + 1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            #new_lr_decay = new_lr_decay * lr_decay
            sess.run(tf.compat.v1.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f\n" % (lr_init * new_lr_decay)
            # print(log)
            with open(checkpoint_dir + "/training_synthetic_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
        elif epoch == tl.global_flag['start_from']:
            sess.run(tf.compat.v1.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f\n" % (lr_init, decay_every, lr_decay)
            # print(log)
            with open(checkpoint_dir + "/training_synthetic_metrics.log", "a") as f:
                # perform file operations
                f.write(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0
        new_batch_size = batch_size  #batchsize 50->40 + 10(augmented)

        #data suffle***
        train_blur_imgs, train_mask_imgs = unison_shuffled_copies(train_blur_imgs, train_mask_imgs)

        for idx in range(0, len(train_blur_imgs) , new_batch_size):
            step_time = time.time()

            augmentation = random.choice(augmentation_list)
            if augmentation == 0:
                imlist = train_blur_imgs[idx: idx + new_batch_size]
                clist = train_mask_imgs[idx: idx + new_batch_size]
            elif augmentation == 1:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],
                                 train_mask_imgs[idx: idx + new_batch_size])],fn=crop_sub_img_and_classification_fn_aug)

                # print images_and_score.shape
                imlist, clist = images_and_score.transpose((1, 0, 2, 3, 4))
                # print clist.shape
                clist = clist[:, :, :, 0]
                # print clist.shape
                clist = np.expand_dims(clist, axis=3)

            #print imlist.shape, clist.shape
            err,l1,l2,l3,l4, _ ,outmap = sess.run([loss,loss1,loss2,loss3,loss4, train_op,out], {patches_blurred: imlist,
                                                                                            classification_map: clist})

            outmap1 = np.squeeze(outmap[1,:,:,0])
            outmap2 = np.squeeze(outmap[1, :, :, 1])
            outmap3 = np.squeeze(outmap[1, :, :, 2])

            if(idx%100 ==0):
                scipy.misc.imsave(save_dir_sample + '/input_mask.png', np.squeeze(clist[1, :, :, 0]))
                scipy.misc.imsave(save_dir_sample + '/input.png', np.squeeze(imlist[1,:,:,:]))
                scipy.misc.imsave(save_dir_sample + '/im.png', outmap1)
                scipy.misc.imsave(save_dir_sample + '/im1.png', outmap2)
                scipy.misc.imsave(save_dir_sample + '/im2.png', outmap3)

            #metrics_file.write("Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3:
            # %.6f,loss4: %.6f" % (epoch, n_epoch, n_iter, time.time() - step_time, err,l1,l2,l3,l4))
            total_loss += err
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f\n" % (epoch, n_epoch, time.time() - epoch_time,
                                                                      total_loss/n_iter)
        # only way to write to log file while running
        with open(checkpoint_dir + "/training_synthetic_metrics.log", "a") as f:
            # perform file operations
            f.write(log)

        ## save model
        if epoch % 10 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='final_SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                            save_dir = checkpoint_dir, var_list = a_vars, global_step = epoch, printable = False)

# my new code for training the data using this version.
# TODO Remove when we verify the updated version of the code
def train_with_ssc_dataset():
    checkpoint_dir = "test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    save_dir_sample = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    input_path = config.TRAIN.ssc_blur_path
    gt_path = config.TRAIN.ssc_gt_path
    train_blur_img_list = sorted(tl.files.load_file_list(path=input_path, regx='/*.(png|PNG)',printable=False))
    train_mask_img_list = sorted(tl.files.load_file_list(path=gt_path, regx='/*.(png|PNG)',printable=False))

    ###Load Training Data ####
    train_blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=100, mode='RGB')
    train_mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=100, mode='RGB2GRAY')

    train_classification_mask = []
    # print train_mask_imgs
     # img_n = 0
    for img in train_mask_imgs:
        tmp_class = img
        tmp_classification = np.concatenate((img, img, img), axis=2)

        tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
        tmp_class[np.where(tmp_classification[:, :, 0] == 64)] = 1  # motion blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 128)] = 2  # out of focus blur
        if np.where(tmp_classification[:, :, 0] == 192)[0].size == 0:
            tmp_class[np.where(tmp_classification[:, :, 0] == 255)] = 4  # brightness blur
        else:
            tmp_class[np.where(tmp_classification[:, :, 0] == 192)] = 3  # darkness blur

        train_classification_mask.append(tmp_class)

    train_mask_imgs = train_classification_mask
    print(len(train_blur_imgs), len(train_mask_imgs))

    ### DEFINE MODEL ###
    # gpu allocation
    # device_type = 'GPU'
    # devices = tf.config.experimental.list_physical_devices(
    #     device_type)
    # devices_names = [d.name.split("e:")[1]for d in devices]
    # strategy = tf.distribute.MirroredStrategy(devices=devices_names,cross_device_ops=tf.distribute.ReductionToOneDevice())
    patches_blurred = tf.compat.v1.placeholder('float32', [batch_size, h, w, 3], name='input_patches')
    # labels_sigma = tf.compat.v1.placeholder('float32', [batch_size,h,w, 1], name = 'lables')
    classification_map = tf.compat.v1.placeholder('int32', [batch_size, h, w, 1], name='labels')
    #with strategy.scope():
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3 = Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3, reuse=False,
                                                                    scope=scope2)

    ### DEFINE LOSS ###
    loss1 = tl.cost.cross_entropy((net_regression.outputs), tf.squeeze(classification_map), name='loss1')
    loss2 = tl.cost.cross_entropy((m1.outputs),
                                      tf.squeeze(tf.image.resize(classification_map, [int(h / 2), int(w / 2)],
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                      name='loss2')
    loss3 = tl.cost.cross_entropy((m2.outputs),
                                      tf.squeeze(tf.image.resize(classification_map, [int(h / 4), int(w / 4)],
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                      name='loss3')
    loss4 = tl.cost.cross_entropy((m3.outputs),
                                      tf.squeeze(tf.image.resize(classification_map, [int(h / 8), int(w / 8)],
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                      name='loss4')
    out = (net_regression.outputs)
    loss = loss1 + loss2 + loss3 + loss4

    with tf.compat.v1.variable_scope('learning_rate'):
        #lr_v = tf.Variable(lr_init * 0.1 * 0.1, trainable=False)
        lr_v2 = tf.Variable(lr_init * 0.1, trainable=False)

    ### DEFINE OPTIMIZER ###
    a_vars = tl.layers.get_variables_with_name('', False, True)  # Unified
    #var_list1 = tl.layers.get_variables_with_name('VGG', True, True)  # ?
    var_list2 = tl.layers.get_variables_with_name('UNet', True, True)  # ?
    opt2 = tf.optimizers.Adam(learning_rate=lr_v2)
    grads = tf.gradients(ys=loss, xs=var_list2, unconnected_gradients='zero')
    train_op2 = opt2.apply_gradients(zip(grads, var_list2))
    train_op = tf.group(train_op2)
    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    # # gpus = tf.config.experimental.list_physical_devices('GPU')
    # # if gpus:
    # #     try:
    # #         for gpu in gpus:
    # #             tf.config.experimental.set_memory_growth(gpu, True)
    #
    # #     except RuntimeError as e:
    # #         print(e)
    sess = tf.compat.v1.Session(config=configTf)

    print("initializing global variable...")
    sess.run(tf.compat.v1.global_variables_initializer())
    print("initializing global variable...DONE")

    ### initalize weights ###
    # initialize weights from previous run if indicated
    if tl.global_flag['start_from'] != 0:
        print("loading initial checkpoint")
        tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                           save_dir=checkpoint_dir, var_list=a_vars, is_latest=True, printable=True)
    else:
        # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
        file_name = 'SA_net_ssc_smaller_dataset_256.ckpt-500'
        reader = py_checkpoint_reader.NewCheckpointReader(file_name)

        state_dict = {
            v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
        }
        # print(tf.trainable_variables())
        get_weights(sess, net_regression, state_dict)
        #get_weights(sess,n)

    ### START TRAINING ###
    augmentation_list = [0, 1]
    train_blur_imgs = np.array(train_blur_imgs, dtype=object)
    train_classification_mask = np.array(train_classification_mask, dtype=object)
    net_regression.eval()
    for epoch in range(tl.global_flag['start_from'], n_epoch + 1):
        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            # new_lr_decay = lr_decay ** (epoch // decay_every)
            # new_lr_decay = new_lr_decay * lr_decay
            #sess.run(tf.compat.v1.assign(lr_v, lr_v * lr_decay))
            sess.run(tf.compat.v1.assign(lr_v2, lr_v2 * lr_decay))
            #
            log = " ** new learning rate for Decoder %f\n" % (sess.run(lr_v2))
            # print(log)
            with open(checkpoint_dir + "/training_CHUK_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
        elif epoch == tl.global_flag['start_from']:
            log = " ** init lr for Encoder: %f decay_every_init: %d, lr_decay: %f\n" % (sess.run(lr_v2),decay_every,
                                                                                                    lr_decay)
            # print(log)
            with open(checkpoint_dir + "/training_CHUK_metrics.log", "a") as f:
                # perform file operations
                f.write(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0
        new_batch_size = batch_size  # batchsize 50->40 + 10(augmented)

        # data suffle***
        train_blur_imgs, train_classification_mask = unison_shuffled_copies(train_blur_imgs, train_classification_mask)

        for idx in range(0, len(train_blur_imgs), new_batch_size):
            step_time = time.time()

            # augmentation_list = [0, 1]
            augmentation = random.choice(augmentation_list)
            if augmentation == 0:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],
                                                                            train_classification_mask[
                                                                            idx: idx + new_batch_size])],
                                                            fn=crop_sub_img_and_classification_fn)
            elif augmentation == 1:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],
                                                                            train_classification_mask[
                                                                            idx: idx + new_batch_size])],
                                                            fn=crop_sub_img_and_classification_fn_aug)
            # print images_and_score.shape
            imlist, clist = images_and_score.transpose((1, 0, 2, 3, 4))
            # print clist.shape
            clist = clist[:, :, :, 0]
            # print clist.shape
            clist = np.expand_dims(clist, axis=3)

            err, l1, l2, l3, l4, _, outmap = sess.run([loss, loss1, loss2, loss3, loss4, train_op, out],
                                                      {net_regression.inputs: imlist,
                                                       classification_map: clist})
            #
            # err = strategy.run(train_step, args=(imlist,clist,net_regression,m1,m2,m3,opt2,distributed_values,))#sess.run(train_step(imlist,clist,net_regression,m1,m2,m3,opt2,distributed_values))

            # outmap1 = np.squeeze(outmap[1,:,:,0])
            # outmap2 = np.squeeze(outmap[1, :, :, 1])
            # outmap3 = np.squeeze(outmap[1, :, :, 2])

            # if(idx%100 ==0):
            # cv2.imwrite(save_dir_sample + '/input_mask.png', np.squeeze(clist[1, :, :, 0]))
            # cv2.imwrite(save_dir_sample + '/input.png', np.squeeze(imlist[1,:,:,:]))
            # cv2.imwrite(save_dir_sample + '/im.png', outmap1)
            # cv2.imwrite(save_dir_sample + '/im1.png', outmap2)
            # cv2.imwrite(save_dir_sample + '/im2.png', outmap3)
            # https://matthew-brett.github.io/teaching/string_formatting.html
            # print(
            #     "Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3: %.6f,loss4: %.6f" % (
            #     epoch, n_epoch, n_iter, time.time() - step_time, err, l1, l2, l3, l4))
            # metrics_file.write("Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3: %.6f,loss4:
            # %.6f" % (epoch, n_epoch, n_iter, time.time() - step_time, err,l1,l2,l3,l4))
            total_loss += err
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f\n" % (epoch, n_epoch, time.time() - epoch_time,
                                                                        total_loss / n_iter)
        # only way to write to log file while running
        with open(checkpoint_dir + "/training_CHUK_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        ## save model
        if epoch % 10 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                               save_dir=checkpoint_dir, var_list=a_vars, global_step=epoch, printable=False)

def make_dataset(images, labels, num_epochs=1, shuffle_data_seed=0):
    img = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(images,dtype=np.float32))
    lab = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(labels,dtype=np.int32))
    dataset = tf.data.Dataset.zip((img, lab))
    dataset = dataset.repeat(num_epochs).shuffle(buffer_size=100000,seed=shuffle_data_seed,reshuffle_each_iteration=True)
    return dataset

def build_validation(x, y_):
    net_regression.eval()
    m1.eval()
    m2.eval()
    m3.eval()
    output = net_regression(x)
    m1o = m1(x)
    m2o = m2(x)
    m3o = m3(x)
    y = tf.cast(y_,tf.int32)

    loss1 = tl.cost.cross_entropy((output), tf.squeeze(y), name='loss1')
    loss2 = tl.cost.cross_entropy((m1o), tf.squeeze(tf.image.resize(y, [int(h / 2), int(w / 2)],
                                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss2')
    loss3 = tl.cost.cross_entropy((m2o), tf.squeeze(tf.image.resize(y, [int(h / 4), int(w / 4)],
                                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss3')
    loss4 = tl.cost.cross_entropy((m3o), tf.squeeze(tf.image.resize(y, [int(h / 8), int(w / 8)],
                                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss4')
    cost = loss1 + loss2 + loss3 + loss4
    accuracy = tf.cast(tf.math.reduce_sum(1 - tf.math.abs(tf.math.subtract(net_regression.outputs, y_))),
                       dtype=tf.float32) * (1 / (256 * 256))
    return output, [cost, accuracy]

def build_train(x, y_):
    net_regression.train()
    m1.train()
    m2.train()
    m3.train()
    output = net_regression(x)
    m1o = m1(x)
    m2o = m2(x)
    m3o = m3(x)
    y = tf.cast(y_,tf.int32)

    loss1 = tl.cost.cross_entropy((output), tf.squeeze(y), name='loss1')
    loss2 = tl.cost.cross_entropy((m1o),tf.squeeze(tf.image.resize(y, [int(h / 2), int(w / 2)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss2')
    loss3 = tl.cost.cross_entropy((m2o),tf.squeeze(tf.image.resize(y, [int(h / 4), int(w / 4)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss3')
    loss4 = tl.cost.cross_entropy((m3o),tf.squeeze(tf.image.resize(y, [int(h / 8), int(w / 8)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss4')
    cost = loss1 + loss2 + loss3 + loss4
    accuracy = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(net_regression.outputs, y_))),
                    dtype=tf.float32)*(1/(256*256))
    log_tensors = {'cost': cost, 'accuracy': accuracy}
    return output, cost, log_tensors

def UPDATED_train_with_ssc_dataset():
    checkpoint_dir = "test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    save_dir_sample = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    input_path = config.TRAIN.ssc_blur_path
    gt_path = config.TRAIN.ssc_gt_path
    train_blur_img_list = sorted(tl.files.load_file_list(path=input_path, regx='/*.(png|PNG)',printable=False))
    train_mask_img_list = sorted(tl.files.load_file_list(path=gt_path, regx='/*.(png|PNG)',printable=False))

    ###Load Training Data ####
    train_blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=100, mode='RGB')
    train_mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=100, mode='RGB2GRAY')

    train_classification_mask = []
    # print train_mask_imgs
     # img_n = 0
    for img in train_mask_imgs:
        tmp_class = img
        tmp_classification = np.concatenate((img, img, img), axis=2)

        tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
        tmp_class[np.where(tmp_classification[:, :, 0] == 64)] = 1  # motion blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 128)] = 2  # out of focus blur
        if np.where(tmp_classification[:, :, 0] == 192)[0].size == 0:
            tmp_class[np.where(tmp_classification[:, :, 0] == 255)] = 4  # brightness blur
        else:
            tmp_class[np.where(tmp_classification[:, :, 0] == 192)] = 3  # darkness blur

        train_classification_mask.append(tmp_class)

    train_mask_imgs = train_classification_mask
    print(len(train_blur_imgs), len(train_mask_imgs))

    ### DEFINE MODEL ###
    # gpu allocation
    # device_type = 'GPU'
    # devices = tf.config.experimental.list_physical_devices(
    #     device_type)
    # devices_names = [d.name.split("e:")[1]for d in devices]
    # strategy = tf.distribute.MirroredStrategy(devices=devices_names,cross_device_ops=tf.distribute.ReductionToOneDevice())
    patches_blurred = tf.compat.v1.placeholder('float32', [batch_size, h, w, 3], name='input_patches')
    # labels_sigma = tf.compat.v1.placeholder('float32', [batch_size,h,w, 1], name = 'lables')
    classification_map = tf.compat.v1.placeholder('int32', [batch_size, h, w, 1], name='labels')
    #with strategy.scope():
    global net_regression
    global m1
    global m3
    global m2
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3 = Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3, reuse=False,
                                                                    scope=scope2)

    ### DEFINE LOSS ###
    loss1 = tl.cost.cross_entropy((net_regression.outputs), tf.squeeze(classification_map), name='loss1')
    loss2 = tl.cost.cross_entropy((m1.outputs),
                                      tf.squeeze(tf.image.resize(classification_map, [int(h / 2), int(w / 2)],
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                      name='loss2')
    loss3 = tl.cost.cross_entropy((m2.outputs),
                                      tf.squeeze(tf.image.resize(classification_map, [int(h / 4), int(w / 4)],
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                      name='loss3')
    loss4 = tl.cost.cross_entropy((m3.outputs),
                                      tf.squeeze(tf.image.resize(classification_map, [int(h / 8), int(w / 8)],
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                      name='loss4')
    out = (net_regression.outputs)
    loss = loss1 + loss2 + loss3 + loss4

    with tf.compat.v1.variable_scope('learning_rate'):
    #     #lr_v = tf.Variable(lr_init * 0.1 * 0.1, trainable=False)
         lr_v2 = tf.Variable(lr_init * 0.1, trainable=False)
    #
    # ### DEFINE OPTIMIZER ###
    a_vars = tl.layers.get_variables_with_name('', False, True)  # Unified
    # #var_list1 = tl.layers.get_variables_with_name('VGG', True, True)  # ?
    var_list2 = tl.layers.get_variables_with_name('UNet', True, True)  # ?
    opt2 = tf.optimizers.Adam(learning_rate=lr_v2)
    grads = tf.gradients(ys=loss, xs=var_list2, unconnected_gradients='zero')
    train_op2 = opt2.apply_gradients(zip(grads, var_list2))
    #train_op = tf.group(train_op2)
    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)

    print("initializing global variable...")
    sess.run(tf.compat.v1.global_variables_initializer())
    print("initializing global variable...DONE")

    ### initalize weights ###
    # initialize weights from previous run if indicated
    if tl.global_flag['start_from'] != 0:
        print("loading initial checkpoint")
        tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                           save_dir=checkpoint_dir, var_list=a_vars, is_latest=True, printable=True)
    else:
        # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
        file_name = 'SA_net_ssc_smaller_dataset_256.ckpt-500'
        reader = py_checkpoint_reader.NewCheckpointReader(file_name)

        state_dict = {
            v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
        }
        # print(tf.trainable_variables())
        get_weights(sess, net_regression, state_dict)

    ### START TRAINING ###
    # augmentation_list = [0, 1]
    train_blur_imgs = np.array(train_blur_imgs, dtype=object)
    train_classification_mask = np.array(train_classification_mask, dtype=object)
    train_blur_imgs, train_classification_mask = unison_shuffled_copies(train_blur_imgs, train_classification_mask)
    numTest = int(.5*len(train_blur_imgs)) # .1 TODO change this
    valid_blur_imgs = train_blur_imgs[:numTest]
    valid_classification_mask = train_classification_mask[:numTest]
    train_blur_imgs = train_blur_imgs[numTest:]
    train_classification_mask = train_classification_mask[numTest:]
    # Setup the trainer
    training_dataset = make_dataset(train_blur_imgs, train_classification_mask,num_epochs=n_epoch)
    training_dataset = training_dataset.map(data_aug_train, num_parallel_calls=multiprocessing.cpu_count())
    validation_dataset = make_dataset(valid_blur_imgs, valid_classification_mask)
    validation_dataset = validation_dataset.map(data_aug_valid, num_parallel_calls=multiprocessing.cpu_count())

    trainer = tl.distributed.Trainer(
            build_training_func=build_train, training_dataset=training_dataset, optimizer=tf.compat.v1.train.AdamOptimizer,
            optimizer_args={'learning_rate':lr_init},batch_size=10, prefetch_size=10,log_step_size=756,
        checkpoint_dir=checkpoint_dir,validation_dataset=validation_dataset, build_validation_func=build_validation )

    # There are multiple ways to use the trainer:
    # 1. Easiest way to train all data: trainer.train_to_end()
    # 2. Train with validation in the middle: trainer.train_and_validate_to_end(validate_step_size=100)
    # 3. Train with full control like follows:
    while not trainer.session.should_stop():
        try:
            # Run a training step synchronously.
            #trainer.train_on_batch()
            trainer.train_and_validate_to_end(validate_step_size=100)
            # TODO: do whatever you like to the training session.
        except tf.errors.OutOfRangeError:
            # The dataset would throw the OutOfRangeError when it reaches the end
            break

    # for epoch in range(tl.global_flag['start_from'], n_epoch + 1):
    #     # update learning rate
    #     # if epoch != 0 and (epoch % decay_every == 0):
    #     #     # new_lr_decay = lr_decay ** (epoch // decay_every)
    #     #     # new_lr_decay = new_lr_decay * lr_decay
    #     #     #sess.run(tf.compat.v1.assign(lr_v, lr_v * lr_decay))
    #     #     sess.run(tf.compat.v1.assign(lr_v2, lr_v2 * lr_decay))
    #     #     #
    #     #     log = " ** new learning rate for Decoder %f\n" % (sess.run(lr_v2))
    #     #     # print(log)
    #     #     with open(checkpoint_dir + "/training_CHUK_metrics.log", "a") as f:
    #     #         # perform file operations
    #     #         f.write(log)
    #     # elif epoch == tl.global_flag['start_from']:
    #     #     log = " ** init lr for Encoder: %f decay_every_init: %d, lr_decay: %f\n" % (sess.run(lr_v2),decay_every,
    #     #                                                                                             lr_decay)
    #     #     # print(log)
    #     #     with open(checkpoint_dir + "/training_CHUK_metrics.log", "a") as f:
    #     #         # perform file operations
    #     #         f.write(log)
    #
    #     epoch_time = time.time()
    #     total_loss, n_iter = 0, 0
    #     new_batch_size = batch_size  # batchsize 50->40 + 10(augmented)
    #
    #     # data suffle***
    #     train_blur_imgs, train_classification_mask = unison_shuffled_copies(train_blur_imgs, train_classification_mask)
    #
    #     for idx in range(0, len(train_blur_imgs), new_batch_size):
    #         step_time = time.time()
    #
    #         # augmentation_list = [0, 1]
    #         augmentation = random.choice(augmentation_list)
    #         if augmentation == 0:
    #             images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],
    #                                                                         train_classification_mask[
    #                                                                         idx: idx + new_batch_size])],
    #                                                         fn=crop_sub_img_and_classification_fn)
    #         elif augmentation == 1:
    #             images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + new_batch_size],
    #                                                                         train_classification_mask[
    #                                                                         idx: idx + new_batch_size])],
    #                                                         fn=crop_sub_img_and_classification_fn_aug)
    #         # print images_and_score.shape
    #         imlist, clist = images_and_score.transpose((1, 0, 2, 3, 4))
    #         # print clist.shape
    #         clist = clist[:, :, :, 0]
    #         # print clist.shape
    #         clist = np.expand_dims(clist, axis=3)
    #
    #         err, l1, l2, l3, l4, _, outmap = sess.run([loss, loss1, loss2, loss3, loss4, train_op, out],
    #                                                   {net_regression.inputs: imlist,
    #                                                    classification_map: clist})
    #         #
    #         # err = strategy.run(train_step, args=(imlist,clist,net_regression,m1,m2,m3,opt2,distributed_values,))#sess.run(train_step(imlist,clist,net_regression,m1,m2,m3,opt2,distributed_values))
    #
    #         # outmap1 = np.squeeze(outmap[1,:,:,0])
    #         # outmap2 = np.squeeze(outmap[1, :, :, 1])
    #         # outmap3 = np.squeeze(outmap[1, :, :, 2])
    #
    #         # if(idx%100 ==0):
    #         # cv2.imwrite(save_dir_sample + '/input_mask.png', np.squeeze(clist[1, :, :, 0]))
    #         # cv2.imwrite(save_dir_sample + '/input.png', np.squeeze(imlist[1,:,:,:]))
    #         # cv2.imwrite(save_dir_sample + '/im.png', outmap1)
    #         # cv2.imwrite(save_dir_sample + '/im1.png', outmap2)
    #         # cv2.imwrite(save_dir_sample + '/im2.png', outmap3)
    #         # https://matthew-brett.github.io/teaching/string_formatting.html
    #         # print(
    #         #     "Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3: %.6f,loss4: %.6f" % (
    #         #     epoch, n_epoch, n_iter, time.time() - step_time, err, l1, l2, l3, l4))
    #         # metrics_file.write("Epoch [%2d/%2d] %4d time: %4.4fs, err: %.6f, loss1: %.6f,loss2: %.6f,loss3: %.6f,loss4:
    #         # %.6f" % (epoch, n_epoch, n_iter, time.time() - step_time, err,l1,l2,l3,l4))
    #         total_loss += err
    #         n_iter += 1

        # log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f\n" % (epoch, n_epoch, time.time() - epoch_time,
        #                                                                 total_loss / n_iter)
        # # only way to write to log file while running
        # with open(checkpoint_dir + "/training_CHUK_metrics.log", "a") as f:
        #     # perform file operations
        #     f.write(log)
        # ## save model
        # if epoch % 10 == 0:
        #     tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
        #                        save_dir=checkpoint_dir, var_list=a_vars, global_step=epoch, printable=False)