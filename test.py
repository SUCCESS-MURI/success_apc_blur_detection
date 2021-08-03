# coding=utf-8
import copy
import csv
import random

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorlayer as tl
import numpy as np
import math

from tensorflow.python.training import py_checkpoint_reader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

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

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

h = config.TRAIN.height
w = config.TRAIN.width

VGG_MEAN = [103.939, 116.779, 123.68]
g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])

ni = int(math.ceil(np.sqrt(batch_size)))

## IOU in pure numpy
# https://github.com/hipiphock/Mean-IOU-in-Numpy-TensorFlow/blob/master/main.py
def numpy_iou(y_true, y_pred, n_class=5):
    # IOU = TP/(TP+FN+FP)
    IOU = []
    for c in range(n_class):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        n = TP.astype(float)
        d = float(TP + FP + FN + 1e-12)

        iou = np.divide(n, d)
        # since not all classes are present in the iou
        if n == 0.0 and d == 1e-12:
            iou = 1.0
        IOU.append(iou)

    return np.mean(IOU)

def blurmap_3classes_using_numpy_pretrainied_weights(index):
    print("Blurmap Generation")

    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_test'
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

                test_image = sharp_image[0: image_h - (image_h % 16), 0: 0 + image_w - (image_w % 16), :] #/ (255.)
                red = test_image[:,:,0]
                green = test_image[:, :, 1]
                blue = test_image[:, :, 2]
                bgr = np.zeros(test_image.shape)
                bgr[:,:,0]= blue - VGG_MEAN[0]
                bgr[:, :, 1] = green - VGG_MEAN[1]
                bgr[:, :, 2] = red - VGG_MEAN[2]
                #bgr = np.round(bgr).astype(np.float32)
                # Model
                patches_blurred = tf.compat.v1.placeholder('float32', [1, test_image.shape[0], test_image.shape[1], 3], name='input_patches')
                if flag==0:
                    reuse =False
                else:
                    reuse =True

                start_time = time.time()

                with tf.compat.v1.variable_scope('Unified') as scope:
                    with tf.compat.v1.variable_scope('VGG') as scope3:
                        input, n, f0, f0_1, f1_2, f2_3= VGG19_pretrained(patches_blurred, reuse=reuse,scope=scope3)
                    with tf.compat.v1.variable_scope('UNet') as scope1:
                        output,m1,m2,m3= Decoder_Network_classification(input, n, f0, f0_1,
                                                                        f1_2, f2_3, reuse = reuse,
                                                                        scope = scope1)

                #a_vars = tl.layers.get_variables_with_name('Unified', False, True)

                configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                configTf.gpu_options.allow_growth = True
                sess = tf.compat.v1.Session(config=configTf)

                tl.layers.initialize_global_variables(sess)

                # Load checkpoint
                #saver.restore(sess, "./setup/checkpoint/final_checkpoint_tf2.ckpt")
                get_weights(sess, output)
                print("loaded all the weights")
                output_map = tf.nn.softmax(output.outputs)
                output_map1 = tf.nn.softmax(m1.outputs)
                output_map2 = tf.nn.softmax(m2.outputs)
                output_map3 = tf.nn.softmax(m3.outputs)

                start_time = time.time()
                blur_map,_,_,_ = sess.run([output_map,output_map1,output_map2,output_map3],{output.inputs: np.expand_dims(
                    (bgr), axis=0)})
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

def testData(index):
    print("Blurmap Generation and Testing")

    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
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

                test_image = sharp_image[0: image_h-(image_h%16), 0: 0 + image_w-(image_w%16), :]#/(255.)
                red = test_image[:, :, 0]
                green = test_image[:, :, 1]
                blue = test_image[:, :, 2]
                bgr = np.zeros(test_image.shape)
                bgr[:, :, 0] = blue - VGG_MEAN[0]
                bgr[:, :, 1] = green - VGG_MEAN[1]
                bgr[:, :, 2] = red - VGG_MEAN[2]

                # Model
                patches_blurred = tf.compat.v1.placeholder('float32', [1, test_image.shape[0], test_image.shape[1], 3], name='input_patches')
                if flag==0:
                    reuse =False
                else:
                    reuse =True

                start_time = time.time()

                with tf.compat.v1.variable_scope('Unified') as scope:
                    with tf.compat.v1.variable_scope('VGG') as scope1:
                        input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=reuse, scope=scope1)
                    with tf.compat.v1.variable_scope('UNet') as scope2:
                        output, m1, m2, m3 = Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3,reuse=reuse,
                                                                                    scope=scope2)

                    output_map = tf.nn.softmax(output.outputs)
                    output_map1 = tf.nn.softmax(m1.outputs)
                    output_map2 = tf.nn.softmax(m2.outputs)
                    output_map3 = tf.nn.softmax(m3.outputs)

                #a_vars = tl.layers.get_variables_with_name('Unified', False, True)

                saver = tf.compat.v1.train.Saver()
                configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                configTf.gpu_options.allow_growth = True
                sess = tf.compat.v1.Session(config=configTf)
                tl.layers.initialize_global_variables(sess)

                # Load checkpoint
                #saver.restore(sess,'SA_net_{}.ckpt-500'.format(tl.global_flag['mode']))
                # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
                file_name = 'SA_net_{}.ckpt-500'.format(tl.global_flag['mode'])
                reader = py_checkpoint_reader.NewCheckpointReader(file_name)

                state_dict = {
                    v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
                }
                #print(tf.trainable_variables())
                get_weights(sess,output,state_dict)

                start_time = time.time()
                blur_map,o1,o2,o3 = sess.run([output_map,output_map1,output_map2,output_map3],{patches_blurred: np.expand_dims(
                    (bgr), axis=0)})
                #np.argmax(sess.run(tf.nn.softmax(net_regression(imlist.astype(np.float32)))), axis=3)
                blur_map = np.squeeze(np.argmax(blur_map,axis=3))
                o1 = np.squeeze(np.argmax(o1,axis=3))
                o2 = np.squeeze(np.argmax(o2,axis=3))
                o3 = np.squeeze(np.argmax(o3,axis=3))

                # now color code
                rgb_blur_map = np.zeros(test_image.shape)
                rgb_o1 = np.zeros((int(test_image.shape[0] / 2), int(test_image.shape[1] / 2), 3))
                rgb_o2 = np.zeros((int(test_image.shape[0] / 4), int(test_image.shape[1] / 4), 3))
                rgb_o3 = np.zeros((int(test_image.shape[0] / 8), int(test_image.shape[1] / 8), 3))
                # red
                rgb_blur_map[blur_map == 1] = [255,0,0]
                rgb_o1[o1 == 1] = [255, 0, 0]
                rgb_o2[o2 == 1] = [255, 0, 0]
                rgb_o3[o3 == 1] = [255, 0, 0]

                # green
                rgb_blur_map[blur_map == 2] = [0, 255, 0]
                rgb_o1[o1 == 2] = [0, 255, 0]
                rgb_o2[o2 == 2] = [0, 255, 0]
                rgb_o3[o3 == 2] = [0, 255, 0]

                # blue
                rgb_blur_map[blur_map == 3] = [0, 0, 255]
                rgb_o1[o1 == 3] = [0, 0, 255]
                rgb_o2[o2 == 3] = [0, 0, 255]
                rgb_o3[o3 == 3] = [0, 0, 255]

                # pink
                rgb_blur_map[blur_map == 4] = [255, 192, 203]
                rgb_o1[o1 == 4] = [255, 192, 203]
                rgb_o2[o2 == 4] = [255, 192, 203]
                rgb_o3[o3 == 4] = [255, 192, 203]

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
                    cv2.imwrite(save_dir_sample + '/' + image.replace(".jpg", ".png"), rgb_blur_map)
                    cv2.imwrite(save_dir_sample + '/m1_results' + image.replace(".jpg", ".png"), rgb_o1)
                    cv2.imwrite(save_dir_sample + '/m2_results' + image.replace(".jpg", ".png"), rgb_o2)
                    cv2.imwrite(save_dir_sample + '/m3_results' + image.replace(".jpg", ".png"), rgb_o3)

                sess.close()
                flag=1

                print("5.--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                if(i==index+101-1):
                    return 0
        i = i + 1
    return 0

# main test function
def testData_return_error():
    print("Blurmap Testing")

    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample+'/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.ssc_blur_path, regx='/*.(png|PNG)', printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.ssc_gt_path, regx='/*.(png|PNG)', printable=False))

    ###Load Testing Data ####
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.ssc_blur_path, n_threads=100, mode='RGB')
    test_mask_imgs = read_all_imgs(test_mask_img_list, path=config.TEST.ssc_gt_path, n_threads=100, mode='RGB2GRAY')

    test_classification_mask = []
    # print train_mask_imgs
    # img_n = 0
    for img in test_mask_imgs:
        tmp_class = img
        tmp_classification = np.concatenate((img, img, img), axis=2)

        tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
        tmp_class[np.where(tmp_classification[:, :, 0] == 64)] = 1  # motion blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 128)] = 2  # out of focus blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 192)] = 3  # darkness blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 255)] = 4  # brightness blur

        test_classification_mask.append(tmp_class)

    test_mask_imgs = test_classification_mask
    print(len(test_blur_imgs), len(test_mask_imgs))

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [1, h, w, 3], name='input_patches')
    classification_map = tf.compat.v1.placeholder('int64', [1, h, w, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3, m1B, m2B, m3B, m4B = Decoder_Network_classification(input, n, f0, f0_1, f1_2,
                                                                                            f2_3, reuse=False,
                                                                                            scope=scope2)

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs),axis=3),axis=3)
    output = tf.nn.softmax(net_regression.outputs)

    ### DEFINE LOSS ###
    loss1 = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(output_map, classification_map))),
                    dtype=tf.float32)*(1/(h*w))
    #mean_iou, update_op = tf.compat.v1.metrics.mean_iou(labels=classification_map,predictions=predictions,num_classes=5)

    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    # Load checkpoint
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    # file_name = 'SA_net_{}.ckpt-500'.format(tl.global_flag['mode'])
    # reader = py_checkpoint_reader.NewCheckpointReader(file_name)
    #
    # state_dict = {
    #     v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
    # }
    # # print(tf.trainable_variables())
    # get_weights(sess, net_regression, state_dict)
    saver = tf.compat.v1.train.Saver()
    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)

    # Load checkpoint
    saver.restore(sess,'./model/SA_net_{}.ckpt'.format(tl.global_flag['mode']))

    net_regression.test()
    m1.test()
    m2.test()
    m3.test()

    accuracy_list = []
    miou_list = []
    f1score_list = []
    classesList = [[],[],[],[],[]]
    # initalize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1', 'Accuracy for Class 2',
                        'Accuracy for Class 3','Accuracy for Class 4','mIOU','f1_score'])

    for i in range(len(test_blur_imgs)):
        test_image = test_blur_imgs[i]
        gt_test_image = test_mask_imgs[i]
        image_name = test_blur_img_list[i]
        red = test_image[:, :, 0]
        green = test_image[:, :, 1]
        blue = test_image[:, :, 2]
        test_image = np.zeros(test_image.shape)
        test_image[:, :, 0] = blue - VGG_MEAN[0]
        test_image[:, :, 1] = green - VGG_MEAN[1]
        test_image[:, :, 2] = red - VGG_MEAN[2]

        # Model
        start_time = time.time()

        # uncertain labeling
        # step run we run the network 100 times
        blurMap = np.squeeze(sess.run([output],{net_regression.inputs: np.expand_dims((test_image), axis=0)}))
        blur_map = np.zeros((256,256))
        blur_map[np.sum(blurMap[:,:] >= .2,axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:,:] >= .2,axis=2) == 1],
                                                                     axis=1)
        # uncertainty labeling
        blur_map[np.sum(blurMap[:, :] >= .2, axis=2) != 1] = 5

        #np.save(save_dir_sample + '/raw_' + image_name.replace(".png", ".npy"), np.squeeze(blur_map))
        accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(),np.squeeze(blur_map).flatten(),normalize=True)

        # calculate mean intersection of union
        miou = numpy_iou(np.squeeze(gt_test_image),np.squeeze(blur_map))

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        perclass_accuracy_conf_matrix = confusion_matrix(np.squeeze(gt_test_image).flatten(),np.squeeze(blur_map).flatten(),
                                       labels=[0,1,2,3,4],normalize="true")

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(5):
            if (perclass_accuracy_conf_matrix[lab,:] == 0).all() and (perclass_accuracy_conf_matrix[:,lab] == 0).all():
                pass
            else:
                classesList[lab].append(perclass_accuracy[lab])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(np.squeeze(gt_test_image).flatten(),np.squeeze(blur_map).flatten(), labels=[0,1,2,3,4],
                           average='micro')

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        f1score_list.append(f1score)

        blur_map = np.squeeze(blur_map)
        gt_map = np.squeeze(gt_test_image)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape)
        rgb_gt_map = np.zeros(test_image.shape)
        # blue motion blur
        rgb_blur_map[blur_map == 1] = [255,0,0]
        rgb_gt_map[gt_map == 1] = [255,0,0]
        # green focus blur
        rgb_blur_map[blur_map == 2] = [0, 255, 0]
        rgb_gt_map[gt_map == 2] = [0, 255, 0]
        # red darkness blur
        rgb_blur_map[blur_map == 3] = [0, 0, 255]
        rgb_gt_map[gt_map == 3] = [0, 0, 255]
        # pink brightness blur
        rgb_blur_map[blur_map == 4] = [255, 192, 203]
        rgb_gt_map[gt_map == 4] = [255, 192, 203]
        # yellow unknown blur
        rgb_blur_map[blur_map == 5] = [0, 255, 255]

        log = "[*] Testing image name:"+image_name+" time: %4.4fs, Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
                                    "Accuracy for Class 1 %.8f Accuracy for Class 2 %.8f Accuracy for Class 3 %.8f " \
                                    "Accuracy for Class 4 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (time.time() - start_time,accuracy,perclass_accuracy[0],perclass_accuracy[1],perclass_accuracy[2],
                 perclass_accuracy[3],perclass_accuracy[4],miou,f1score)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image_name,str(np.round(accuracy,8)),str(np.round(perclass_accuracy[0],8)),
                   str(np.round(perclass_accuracy[1],8)),str(np.round(perclass_accuracy[2],8)),
                   str(np.round(perclass_accuracy[3],8)),str(np.round(perclass_accuracy[4],8)),str(np.round(miou,8)),
                   str(np.round(f1score,8))]
        with open(save_dir_sample + "/testing_metrics.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(string_list)

        if ".jpg" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)
        if ".JPG" in image_name:
            image_name.replace(".JPG", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".JPG", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".JPG", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".JPG", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".JPG", ".png"), rgb_o3)
        if ".PNG" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)
        if ".png" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)

    sess.close()
    log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f " \
          "Max Accuracy Class 2: %.8f Max Accuracy Class 3: %.8f Max Accuracy Class 4: %.8f Max IoU: %.8f " \
          "Variance: %.8f Max F1_score: %.8f\n" % (np.max(np.array(accuracy_list)),
           np.max(np.array(classesList[0]),axis=0),np.max(np.array(classesList[1]),axis=0),
           np.max(np.array(classesList[2]),axis=0),np.max(np.array(classesList[3]),axis=0),
           np.max(np.array(classesList[4]),axis=0),np.max(np.array(miou_list)),np.var(np.asarray(accuracy_list)),
           np.max(np.array(f1score_list)))
    log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
          "Mean Accuracy Class 2: %.8f Mean Accuracy Class 3: %.8f Mean Accuracy Class 4: %.8f Mean IoU: %.8f " \
          "Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_list)), np.mean(np.array(classesList[0])),
           np.mean(np.array(classesList[1])),np.mean(np.array(classesList[2])),np.mean(np.array(classesList[3])),
           np.mean(np.array(classesList[4])),np.mean(np.array(miou_list)),np.mean(np.array(f1score_list)))
    # only way to write to log file while running
    with open(save_dir_sample + "/testing_metrics.log", "a") as f:
        # perform file operations
        f.write(log)
        f.write(log2)
    return 0

