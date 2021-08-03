# coding=utf-8
import csv
import random

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import math
from sklearn.metrics import confusion_matrix

from config import config, log_config
from utils import read_all_imgs, crop_sub_img_and_classification_fn_aug, crop_sub_img_and_classification_fn
from model import Decoder_Network_classification, VGG19_pretrained
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
do_validation_every = config.TRAIN.valid_every
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

h = config.TRAIN.height
w = config.TRAIN.width

ni = int(math.ceil(np.sqrt(batch_size)))

# https://izziswift.com/better-way-to-shuffle-two-numpy-arrays-in-unison/
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

# get the weights for each model but leave off the last layer for label change
def get_weights(sess,network):
    # https://github.com/TreB1eN/InsightFace_Pytorch/issues/137
    dict_weights_trained = np.load('./setup/final_model.npy',allow_pickle=True)[()]
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

# my new code for training the data using this version.
def train_with_muri_dataset():
    checkpoint_dir = "test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    # get the images for the training and validation #
    input_path = config.TRAIN.muri_blur_path
    gt_path = config.TRAIN.muri_gt_path
    train_blur_img_list = sorted(tl.files.load_file_list(path=input_path, regx='/*.(png|PNG)', printable=False))
    train_mask_img_list = sorted(tl.files.load_file_list(path=gt_path, regx='/*.(png|PNG)', printable=False))
    valid_input_path = config.VALIDATION.muri_blur_path
    valid_gt_path = config.VALIDATION.muri_gt_path
    validation_blur_img_list = sorted(
        tl.files.load_file_list(path=valid_input_path, regx='/*.(png|PNG)', printable=False))
    validation_mask_img_list = sorted(tl.files.load_file_list(path=valid_gt_path, regx='/*.(png|PNG)', printable=False))

    ###Load Training Data ####
    train_blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=100, mode='RGB')
    train_mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=100, mode='RGB2GRAY2')
    valid_blur_imgs = read_all_imgs(validation_blur_img_list, path=valid_input_path, n_threads=100, mode='RGB')
    valid_mask_imgs = read_all_imgs(validation_mask_img_list, path=valid_gt_path, n_threads=100, mode='RGB2GRAY2')
    # print train_mask_imgs
    train_classification_mask = []
    # img_n = 0
    for img in train_mask_imgs:
        tmp_class = img
        tmp_classification = np.concatenate((img, img, img), axis=2)

        tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
        tmp_class[np.where(tmp_classification[:, :, 0] == 64)] = 1  # motion blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 128)] = 2  # out of focus blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 192)] = 3  # darkness blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 255)] = 4  # brightness blur

        train_classification_mask.append(tmp_class)

    train_blur_imgs = np.array(train_blur_imgs, dtype=object)
    train_classification_mask = np.array(train_classification_mask, dtype=object)

    valid_classification_mask = []
    # img_n = 0
    for img in valid_mask_imgs:
        tmp_class = img
        tmp_classification = np.concatenate((img, img, img), axis=2)

        tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
        tmp_class[np.where(tmp_classification[:, :, 0] == 64)] = 1  # motion blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 128)] = 2  # out of focus blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 192)] = 3  # darkness blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 255)] = 4  # brightness blur

        valid_classification_mask.append(tmp_class)

    valid_blur_imgs = np.array(valid_blur_imgs, dtype=object)
    valid_classification_mask = np.array(valid_classification_mask, dtype=object)

    print("Number of training images " + str(len(train_blur_imgs)))
    print("Number of validation images " + str(len(valid_blur_imgs)))

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [batch_size, h, w, 3], name='input_patches')
    classification_map = tf.compat.v1.placeholder('int32', [batch_size, h, w, 1], name='labels')
    # with strategy.scope():
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3, m1B, m2B, m3B, m4B = Decoder_Network_classification(input, n, f0, f0_1, f1_2,
                                                                                            f2_3, reuse=False,
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
    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
    loss = loss1 + loss2 + loss3 + loss4

    with tf.compat.v1.variable_scope('learning_rate'):
        # lr_v = tf.Variable(lr_init * 0.1 * 0.1, trainable=False)
        lr_v2 = tf.Variable(lr_init * 0.1, trainable=False)

    ### DEFINE OPTIMIZER ###
    a_vars = tl.layers.get_variables_with_name('', False, True)  # Unified
    var_list2 = tl.layers.get_variables_with_name('UNet', True, True)  # ?
    opt2 = tf.optimizers.Adam(learning_rate=lr_v2)
    grads = tf.gradients(ys=loss, xs=var_list2, unconnected_gradients='zero')
    train_op2 = opt2.apply_gradients(zip(grads, var_list2))
    train_op = tf.group(train_op2)
    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #
    #     except RuntimeError as e:
    #         print(e)
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
        # we are updating all of the pretrained weights up to the last layer
        get_weights(sess, n)
        get_weights(sess, m1B)
        get_weights(sess, m2B)
        get_weights(sess, m3B)
        get_weights(sess, m4B)

    ### START TRAINING ###
    augmentation_list = [0, 1]

    # initialize the csv metrics output
    with open(checkpoint_dir + "/training_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'total_error'])

    # initialize the csv metrics output
    with open(checkpoint_dir + "/validation_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Epoch', 'total_error', 'Accuracy for Class 0', 'Accuracy for Class 1', 'Accuracy for Class 2',
             'Accuracy for Class 3', 'Accuracy for Class 4'])

    for epoch in range(tl.global_flag['start_from'], n_epoch + 1):
        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            # new_lr_decay = lr_decay ** (epoch // decay_every)
            # new_lr_decay = new_lr_decay * lr_decay
            # sess.run(tf.compat.v1.assign(lr_v, lr_v * lr_decay))
            sess.run(tf.compat.v1.assign(lr_v2, lr_v2 * lr_decay))
            #
            log = " ** new learning rate for Decoder %f\n" % (sess.run(lr_v2))
            # print(log)
            with open(checkpoint_dir + "/training_ssc_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
        elif epoch == tl.global_flag['start_from']:
            log = " ** init lr for Decoder: %f decay_every_init: %d, lr_decay: %f\n" % (sess.run(lr_v2), decay_every,
                                                                                        lr_decay)
            # print(log)
            with open(checkpoint_dir + "/training_ssc_metrics.log", "a") as f:
                # perform file operations
                f.write(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0
        new_batch_size = batch_size  # batchsize 50->40 + 10(augmented)

        # data suffle***
        train_blur_imgs, train_classification_mask = unison_shuffled_copies(train_blur_imgs, train_classification_mask)

        for idx in range(0, len(train_blur_imgs), new_batch_size):
            # step_time = time.time()

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

            err,_ = sess.run([loss,train_op,],{net_regression.inputs: imlist,classification_map: clist})
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
        with open(checkpoint_dir + "/training_ssc_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        with open(checkpoint_dir + "/training_metrics.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, str(np.round(total_loss / n_iter, 8))])

        ## save model
        if epoch % 10 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                               save_dir=checkpoint_dir, var_list=a_vars, global_step=epoch, printable=False)

        # check and do validation if necessary
        if epoch != 0 and (epoch % do_validation_every == 0):
            total_loss, n_iter = 0, 0
            new_batch_size = batch_size  # batchsize 50->40 + 10(augmented)
            step_time = time.time()
            classesList = [[],[],[],[],[]]
            # validation just crop
            for idx in range(0, len(valid_blur_imgs), new_batch_size):
                images_and_score = tl.prepro.threading_data(
                    [_ for _ in zip(valid_blur_imgs[idx: idx + new_batch_size],
                                    valid_classification_mask[
                                    idx: idx + new_batch_size])],
                    fn=crop_sub_img_and_classification_fn)

                # print images_and_score.shape
                imlist, clist = images_and_score.transpose((1, 0, 2, 3, 4))
                # print clist.shape
                clist = clist[:, :, :, 0]
                # print clist.shape
                clist = np.expand_dims(clist, axis=3)

                err, outmap = sess.run([loss, output_map],{net_regression.inputs: imlist,classification_map: clist})

                # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
                perclass_accuracy_conf_matrix = confusion_matrix(np.squeeze(clist).flatten(), np.squeeze(outmap).flatten(),
                                                                 labels=[0, 1, 2, 3, 4], normalize="true")

                perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
                for lab in range(5):
                    if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (
                            perclass_accuracy_conf_matrix[:, lab] == 0).all():
                        pass
                    else:
                        classesList[lab].append(perclass_accuracy[lab])
                total_loss += err
                n_iter += 1

            log = "[*] Validation Results: time: %4.4fs, total_err: %.8f Accuracy per class: 0: %.8f, 1: %.8f, 2: %.8f, 3: %.8f, 4: %.8f\n" % (
                time.time() - step_time,
                total_loss / n_iter, np.mean(np.array(classesList[0])), np.mean(np.array(classesList[1])),
                np.mean(np.array(classesList[2])),np.mean(np.array(classesList[3])), np.mean(np.array(classesList[4])))
            # only way to write to log file while running
            with open(checkpoint_dir + "/training_ssc_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
            with open(checkpoint_dir + "/validation_metrics.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [str(epoch), str(np.round(total_loss / n_iter, 8)), str(np.round(np.mean(np.array(classesList[0])), 8)),
                     str(np.round(np.mean(np.array(classesList[1])), 8)), str(np.round(np.mean(np.array(classesList[2])), 8)),
                     str(np.round(np.mean(np.array(classesList[3])), 8)), str(np.round(np.mean(np.array(classesList[4])), 8))])
