# coding=utf-8
import csv
import multiprocessing
import os

import imageio
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

tf.compat.v1.disable_eager_execution()
import tensorlayer as tl
import numpy as np
import math
from tensorflow.python.training import py_checkpoint_reader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from config import config
from utils import read_all_imgs, get_imgs_RGB_fn, get_imgs_RGBGRAY_2_fn
from model import VGG19_pretrained, Decoder_Network_classification_3_labels, \
    Updated_Decoder_Network_classification
import matplotlib
import datetime
import time
import cv2
from numba import cuda
device = cuda.get_current_device()
device.reset()

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

VGG_MEAN = [103.939, 116.779, 123.68]
g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])

threshold = .3

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

def get_weights(sess,network):
    # https://github.com/TreB1eN/InsightFace_Pytorch/issues/137
    dict_weights_trained = np.load('final_model.npy',allow_pickle=True)[()]
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

def blurmap_3classes_using_numpy_pretrainied_weights_newds():
    print("Blurmap Generation")
    final_shape = (480, 640)

    #date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_new_pho_our_dataset_bd'
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample+'/gt')
    tl.files.exists_or_mkdir(save_dir_sample+'/binary_result/')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary_result/gt')
    #Put the input path!
    sharp_path = '/home/mary/code/local_success_dataset/IROS_Dataset/02_22_2022/Testing/images/'#./input'
    test_sharp_img_list = os.listdir(sharp_path)
    test_sharp_img_list.sort()

    sharp_gt_path = '/home/mary/code/local_success_dataset/IROS_Dataset/02_22_2022/Testing/gt/'  # ./input'
    test_sharp_gt_list = os.listdir(sharp_gt_path)
    test_sharp_gt_list.sort()

    # Model
    patches_blurred = tf.compat.v1.placeholder('float32', [1, final_shape[0], final_shape[1], 3],name='input_patches')
    labels = tf.compat.v1.placeholder('int64', [1, final_shape[0], final_shape[1], 1], name='labels')

    with tf.compat.v1.variable_scope('Unified') as scope:
        with tf.compat.v1.variable_scope('VGG') as scope3:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope3)
        with tf.compat.v1.variable_scope('UNet') as scope1:
            net_regression, m1, m2, m3 = Decoder_Network_classification_3_labels(input, n, f0, f0_1, f1_2, f2_3,
                                                                                  reuse=False, scope=scope1)

    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)

    # Load checkpoint
    get_weights(sess, net_regression)
    print("loaded all the weights")

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
    # output_map_1 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m1.outputs), axis=3), axis=3)
    # output_map_2 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m2.outputs), axis=3), axis=3)
    # output_map_3 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m3.outputs), axis=3), axis=3)
    # output = tf.nn.softmax(net_regression.outputs)

    ### DEFINE LOSS ###
    loss1 = tf.cast(tf.math.reduce_sum(1 - tf.math.abs(tf.math.subtract(output_map, labels))),
                    dtype=tf.float32) * (1 / (final_shape[0] * final_shape[1]))

    accuracy_list = []
    miou_list = []
    f1score_list = []
    classesList = [[],[],[]]
    accuracy_list_binary = []
    miou_list_binary = []
    f1score_binary_list = []
    classesListBinary = [[], []]
    # initialize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1',
                         'Accuracy for Class 2','mIOU','f1_score'])
    f = None
    with open(save_dir_sample + "/testing_metrics_binary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1','mIOU',
                         'f1_score'])
    f = None
    writer = None
    all_image_results = []
    all_gt_image_results = []
    all_binary_image_results = []
    all_gt_binary_image_results = []

    for i in range(len(test_sharp_img_list)):
        image = test_sharp_img_list[i]
        gt = test_sharp_gt_list[i]
        sharp = os.path.join(sharp_path, image)
        sharp_image = Image.open(sharp)
        sharp_image.load()

        sharp_gt = os.path.join(sharp_gt_path,gt)
        sharp_gt_image = Image.open(sharp_gt)
        sharp_gt_image.load()

        sharp_image = np.asarray(sharp_image, dtype="float32")
        sharp_gt_image = np.asarray(sharp_gt_image, dtype="int64")
        if len(sharp_gt_image.shape) == 3:
            sharp_gt_image = sharp_gt_image[:,:,0]
        # for the sake of testing we will determine brightness and darkness as out of focus
        sharp_gt_image[sharp_gt_image == 0] = 0  # sharp
        sharp_gt_image[sharp_gt_image == 64] = 1  # motion blur
        sharp_gt_image[sharp_gt_image == 128] = 2  # out of focus blur
        sharp_gt_image[sharp_gt_image == 192] = 0  # darkness blur default no blur
        sharp_gt_image[sharp_gt_image == 255] = 0  # brightness blur

        # if sharp_image.shape[0] > sharp_image.shape[1]:
        #     sharp_image = cv2.rotate(sharp_image,0)
        #     sharp_gt_image = cv2.rotate(sharp_gt_image, 0)
        # y1, y2 = max(0, int((final_shape[0] + 1 - sharp_image.shape[0]) / 2)), \
        #          min(sharp_image.shape[0]+int((final_shape[0] + 1 - sharp_image.shape[0]) / 2),
        #              final_shape[0])
        # x1, x2 = max(0, int((final_shape[1] + 1 - sharp_image.shape[1]) / 2)), \
        #          min(sharp_image.shape[1]+int((final_shape[1] + 1 - sharp_image.shape[1]) / 2),
        #              final_shape[1])
        # y1o, y2o = 0, min(sharp_image.shape[0], final_shape[0])
        # x1o, x2o = 0, min(sharp_image.shape[1], final_shape[1])
        #
        # test_image = np.zeros((final_shape[0], final_shape[1], 3))
        # test_image[y1:y2, x1:x2] = sharp_image[y1o:y2o,x1o:x2o]
        # test_gt_image =  np.zeros((final_shape[0], final_shape[1]))
        # test_gt_image[y1:y2, x1:x2] = sharp_gt_image[y1o:y2o,x1o:x2o]

        if(len(sharp_image.shape)<3):
            sharp_image= np.expand_dims(np.asarray(sharp_image), 3)
            sharp_image=np.concatenate([sharp_image, sharp_image, sharp_image],axis=2)

        if (sharp_image.shape[2] ==4):
            print(sharp_image.shape)
            sharp_image = np.expand_dims(np.asarray(sharp_image), 3)

            print(sharp_image.shape)
            sharp_image = np.concatenate((sharp_image[:,:,0],sharp_image[:,:,1],sharp_image[:,:,2]),axis=2)

        #print(sharp_image.shape)

        image_h, image_w =sharp_image.shape[0:2]
        #print(image_h, image_w)

        test_image = sharp_image[0: image_h - (image_h % 16), 0: 0 + image_w - (image_w % 16), :] #/ (255.)
        test_gt_image = sharp_gt_image[0: image_h - (image_h % 16), 0: 0 + image_w - (image_w % 16)]

        red = test_image[:,:,0]
        green = test_image[:, :, 1]
        blue = test_image[:, :, 2]
        bgr = np.zeros(test_image.shape)
        bgr[:,:,0]= blue - VGG_MEAN[0]
        bgr[:, :, 1] = green - VGG_MEAN[1]
        bgr[:, :, 2] = red - VGG_MEAN[2]

        # need to have blur-no-blur comparison
        test_gt_image_blur_no_blur = np.copy(test_gt_image)
        # only 1 label
        test_gt_image_blur_no_blur[test_gt_image > 0] = 1

        # https://github.com/tensorflow/tensorflow/issues/36465
        # might use this for process above
        #https://newbedev.com/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-process
        # manager = multiprocessing.Manager()
        # return_dict = manager.dict()
        # p = multiprocessing.Process(target=run_session,args=(bgr,test_gt_image,test_gt_image_blur_no_blur,return_dict))
        # p.start()
        # p.join()
        # #run_session(bgr,test_gt_image,return_dict)
        # blur_map,o1,o2,o3,accuracy,blur_map_binary,accuracy_binary = return_dict.values()

        blur_map = sess.run([output_map],{net_regression.inputs: np.expand_dims((bgr), axis=0)})[0]
        accuracy = accuracy_score(np.squeeze(test_gt_image).flatten(), np.squeeze(blur_map).flatten(), normalize=True)
        # compare binary map
        blur_map_binary = np.copy(blur_map)
        blur_map_binary[blur_map > 0] = 1

        accuracy_binary = sess.run([loss1],{output_map: blur_map_binary,
                                    labels: np.expand_dims((test_gt_image_blur_no_blur[:,:,np.newaxis]),axis=0)})[0]

        blur_map = np.squeeze(blur_map)
        blur_map_binary = np.squeeze(blur_map_binary)
        # o1 = np.squeeze(o1)
        # o2 = np.squeeze(o2)
        # o3 = np.squeeze(o3)

        all_image_results.append(blur_map)
        all_gt_binary_image_results.append(test_gt_image_blur_no_blur)
        all_binary_image_results.append(blur_map_binary)
        all_gt_image_results.append(test_gt_image)

        # calculate mean intersection of union
        miou = numpy_iou(test_gt_image, blur_map, 3)
        miou_binary = numpy_iou(test_gt_image_blur_no_blur, blur_map_binary, 2)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        #accuracy0 = accuracy_score(test_gt_image.flatten(),blur_map.flatten(),normalize="true")
        perclass_accuracy_conf_matrix = confusion_matrix(test_gt_image.flatten(),blur_map.flatten(),labels=[0, 1, 2],
                                                         normalize="true")

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(3):
            if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (perclass_accuracy_conf_matrix[:, lab] == 0).all():
                pass
            else:
                classesList[lab].append(perclass_accuracy[lab])

        perclass_accuracy_conf_matrix = confusion_matrix(test_gt_image_blur_no_blur.flatten(), blur_map_binary.flatten(),
                                                         labels=[0, 1],normalize="true")

        perclass_accuracy_binary = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(2):
            if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (
                    perclass_accuracy_conf_matrix[:, lab] == 0).all():
                pass
            else:
                classesListBinary[lab].append(perclass_accuracy_binary[lab])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(test_gt_image.flatten(), blur_map.flatten(),labels=[0, 1, 2],average='micro')
        f1score_binary = f1_score(test_gt_image_blur_no_blur.flatten(), blur_map_binary.flatten(), labels=[0, 1],
                                  average='micro')

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        f1score_list.append(f1score)
        accuracy_list_binary.append(accuracy_binary)
        miou_list_binary.append(miou_binary)
        f1score_binary_list.append(f1score_binary)

        # now color code
        # now color code
        rgb_blur_map = np.zeros(test_image.shape).astype(np.uint8)
        # rgb_blur_map_1 = np.zeros((o1.shape[0], o1.shape[1], 3)).astype(np.uint8)
        # rgb_blur_map_2 = np.zeros((o2.shape[0], o2.shape[1], 3)).astype(np.uint8)
        # rgb_blur_map_3 = np.zeros((o3.shape[0], o3.shape[1], 3)).astype(np.uint8)
        rgb_gt_map = np.zeros(test_image.shape).astype(np.uint8)
        # rgb_gt_map_1 = np.zeros((o1.shape[0], o1.shape[1], 3)).astype(np.uint8)
        # rgb_gt_map_2 = np.zeros((o2.shape[0], o2.shape[1], 3)).astype(np.uint8)
        # rgb_gt_map_3 = np.zeros((o3.shape[0], o3.shape[1], 3)).astype(np.uint8)
        # gt_o1 = np.squeeze(cv2.resize(test_gt_image, (o1.shape[1], o1.shape[0]), interpolation=cv2.INTER_NEAREST))
        # gt_o2 = np.squeeze(cv2.resize(test_gt_image, (o2.shape[1], o2.shape[0]), interpolation=cv2.INTER_NEAREST))
        # gt_o3 = np.squeeze(cv2.resize(test_gt_image, (o3.shape[1], o3.shape[0]), interpolation=cv2.INTER_NEAREST))
        # red motion blur
        rgb_blur_map[blur_map == 1] = [255, 0, 0]
        # rgb_blur_map_1[o1 == 1] = [255, 0, 0]
        # rgb_blur_map_2[o2 == 1] = [255, 0, 0]
        # rgb_blur_map_3[o3 == 1] = [255, 0, 0]
        rgb_gt_map[test_gt_image == 1] = [255, 0, 0]
        # rgb_gt_map_1[gt_o1 == 1] = [255, 0, 0]
        # rgb_gt_map_2[gt_o2 == 1] = [255, 0, 0]
        # rgb_gt_map_3[gt_o3 == 1] = [255, 0, 0]
        # green focus blur
        rgb_blur_map[blur_map == 2] = [0, 255, 0]
        # rgb_blur_map_1[o1 == 2] = [0, 255, 0]
        # rgb_blur_map_2[o2 == 2] = [0, 255, 0]
        # rgb_blur_map_3[o3 == 2] = [0, 255, 0]
        rgb_gt_map[test_gt_image == 2] = [0, 255, 0]
        # rgb_gt_map_1[gt_o1 == 2] = [0, 255, 0]
        # rgb_gt_map_2[gt_o2 == 2] = [0, 255, 0]
        # rgb_gt_map_3[gt_o3 == 2] = [0, 255, 0]

        image_name = image
        if ".jpg" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m1_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m2_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m3_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".JPG" in image_name:
            image_name.replace(".JPG", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".JPG", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map)
            # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_1)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m1_results' + image_name.replace(".JPG", ".png"), rgb_gt_map_1)
            # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_2)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m2_results' + image_name.replace(".JPG", ".png"), rgb_gt_map_2)
            # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_3)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m3_results' + image_name.replace(".JPG", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".JPG", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".JPG", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".PNG" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m1_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m2_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m3_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".png" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m1_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m2_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            # imageio.imwrite(save_dir_sample + '/gt/gt_m3_results' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)

        log = "[*] Testing image name:" + image + " Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
             "Accuracy for Class 1 %.8f Accuracy for Class 2 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (accuracy, perclass_accuracy[0], perclass_accuracy[1], perclass_accuracy[2], miou, f1score)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics.log", "a") as f:
            # perform file operations
            f.write(log)

        log = "[*] Testing image name:" + image + " Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
                                                  "Accuracy for Class 1 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (accuracy_binary, perclass_accuracy_binary[0], perclass_accuracy_binary[1], miou_binary, f1score_binary)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics_binary.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image, str(np.round(accuracy, 8)), str(np.round(perclass_accuracy[0], 8)),
                       str(np.round(perclass_accuracy[1], 8)), str(np.round(perclass_accuracy[2], 8)),
                       str(np.round(miou, 8)),str(np.round(f1score, 8))]
        with open(save_dir_sample + "/testing_metrics.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(string_list)
        # write csv file output for plots making
        string_list = [image, str(np.round(accuracy_binary, 8)), str(np.round(perclass_accuracy_binary[0], 8)),
                           str(np.round(perclass_accuracy_binary[1], 8)),str(np.round(miou, 8)),
                       str(np.round(f1score, 8))]
        with open(save_dir_sample + "/testing_metrics_binary.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(string_list)
        #flag=1
        # patches_blurred,labels,net_regression,input,n,f0, f0_1, f1_2, f2_3,m1,m2,m3,output_map,loss1,sess,scope,scope1,\
        # scope3,configTf,blur_map,bgr,rgb_blur_map,rgb_gt_map,f1score,miou,perclass_accuracy,writer,f,string_list,log,\
        # perclass_accuracy_conf_matrix,accuracy0,red,green,blue = None,None,None,None,None,None,None,None,None,None,None,\
        #                                                          None,None,None,None,None,None,None,None,None,None,None,\
        #                                                          None,None,None,None,None,None,None,None,None,None,None,\
        #                                                          None,None

    log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f " \
          "Max Accuracy Class 2: %.8f Max IoU: %.8f Variance: %.8f Max F1_score: %.8f\n" % (np.max(np.array(accuracy_list)),
           np.max(np.array(classesList[0]),axis=0),np.max(np.array(classesList[1]),axis=0),
           np.max(np.array(classesList[2]),axis=0),np.max(np.array(miou_list)),np.var(np.asarray(accuracy_list)),
           np.max(np.array(f1score_list)))
    log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
          "Mean Accuracy Class 2: %.8f Mean IoU: %.8f Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_list)),
           np.mean(np.array(classesList[0])),np.mean(np.array(classesList[1])),np.mean(np.array(classesList[2])),
           np.mean(np.array(miou_list)),np.mean(np.array(f1score_list)))
    # only way to write to log file while running
    with open(save_dir_sample + "/testing_metrics.log", "a") as f:
        # perform file operations
        f.write(log)
        f.write(log2)

    log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f Max IoU: %.8f " \
          "Variance: %.8f Max F1_score: %.8f\n" % (
          np.max(np.array(accuracy_list_binary)),
          np.max(np.array(classesListBinary[0]), axis=0), np.max(np.array(classesListBinary[1]), axis=0),
          np.max(np.array(miou_list_binary)), np.var(np.asarray(accuracy_list_binary)),
          np.max(np.array(f1score_binary_list)))
    log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
           "Mean IoU: %.8f Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_list_binary)),
                                                                                 np.mean(np.array(classesListBinary[0])),
                                                                                 np.mean(np.array(classesListBinary[1])),
                                                                                 np.mean(np.array(miou_list_binary)),
                                                                                 np.mean(np.array(f1score_binary_list)))
    # only way to write to log file while running
    with open(save_dir_sample + "/testing_metrics_binary.log", "a") as f:
        # perform file operations
        f.write(log)
        f.write(log2)
    f = None

    plt.rc('font', size=20)  # controls default text size
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=30)  # fontsize of the legend

    plt.clf()
    final_confmatrix = confusion_matrix(np.array(all_gt_image_results).flatten(), np.array(all_image_results).flatten(),
                                        labels=[0, 1, 2], normalize="true")
    np.save(save_dir_sample + '/' +'kimweights_all_labels_results_conf_matrix.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix,3)
    plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Motion', 'Focus']
    plt.title('Kim et al. Weights - Test Data Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    thresh = final_confmatrix.max() / 2.
    for i in range(final_confmatrix.shape[0]):
        for j in range(final_confmatrix.shape[1]):
            plt.text(j, i, format(final_confmatrix[i, j]),
                     ha="center", va="center",
                     color="white" if final_confmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(save_dir_sample + '/' +'conf_matrix_all_labels.png')
    plt.show()

    plt.clf()
    final_confmatrix = confusion_matrix(np.array(all_gt_binary_image_results).flatten(),
                                        np.array(all_binary_image_results).flatten(), labels=[0, 1], normalize="true")
    np.save(save_dir_sample + '/' +'kimweights_all_binary_results_conf_matrix.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix, 3)
    plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Blur']
    plt.title('Kim et al. Weights - Test Data Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    # for i in range(len(classNames)):
    #     for j in range(len(classNames)):
    #         plt.text(j, i, str(final_confmatrix[i][j]))
    thresh = final_confmatrix.max() / 2.
    for i in range(final_confmatrix.shape[0]):
        for j in range(final_confmatrix.shape[1]):
            plt.text(j, i, format(final_confmatrix[i, j]),
                     ha="center", va="center",
                     color="white" if final_confmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(save_dir_sample + '/' +'conf_matrix_binary.png')
    plt.show()

    return 0

# main test function
def test_with_muri_dataset():
    print("MURI Testing")

    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample + '/gt')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_blur_path, regx='/*.(png|PNG)',
                                                        printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_gt_path, regx='/*.(png|PNG)',
                                                        printable=False))

    ###Load Testing Data ####
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.muri_blur_path, n_threads=100, mode='RGB')
    test_mask_imgs = read_all_imgs(test_mask_img_list, path=config.TEST.muri_gt_path, n_threads=100, mode='RGB2GRAY2')

    test_classification_mask = []
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

    h = test_blur_imgs[0].shape[0]
    w = test_blur_imgs[0].shape[1]

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [1, h, w, 3], name='input_patches')
    classification_map = tf.compat.v1.placeholder('int64', [1, h, w, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3, m1B, m2B, m3B, m4B = Updated_Decoder_Network_classification(input, n, f0, f0_1, f1_2,
                                                                                            f2_3, reuse=False,
                                                                                            scope=scope2)

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
    output = tf.nn.softmax(net_regression.outputs)

    ### DEFINE LOSS ###
    accuracy_run = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(output_map, classification_map))),
                    dtype=tf.float32)*(1/(h*w))

    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Load checkpoint
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    file_name = './model/SA_net_{}.ckpt'.format(tl.global_flag['mode'])
    reader = py_checkpoint_reader.NewCheckpointReader(file_name)
    state_dict = {v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()}
    get_weights_checkpoint(sess, net_regression, state_dict)

    accuracy_list = []
    miou_list = []
    miou_binary_list = []
    f1score_list = []
    classesListBinary = [[],[],[],[],[]]
    classesList = [[], [], [], [], []]
    accuracy_binary_list = []
    f1score_binary_list = []
    all_gt_image_results = []
    all_image_results = []
    all_gt_binary_image_results = []
    all_binary_image_results = []
    # initalize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1',
                         'Accuracy for Class 2','Accuracy for Class 3','Accuracy for Class 4','mIOU','f1_score'])

    with open(save_dir_sample + "/testing_metrics_bnb.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1'])

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
        gt_blur_no_blur = np.copy(gt_test_image)
        gt_blur_no_blur[gt_test_image > 0] = 1
        # uncertain labeling
        if tl.global_flag['uncertainty_label']:
            blur_map = np.zeros((h, w))
            blurMap = np.squeeze(sess.run([output], {net_regression.inputs: np.expand_dims((test_image), axis=0)}))
            # numpy array with labels
            blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:, :] >=
                                                                        threshold,axis=2) == 1],axis=1)
            # uncertainty labeling
            blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) != 1] = 5
        else:
            blurMap = sess.run([output_map], {net_regression.inputs: np.expand_dims(test_image, axis=0)})[0]
            blur_map = np.squeeze(blurMap)
        blurMap_blurnoblur = np.copy(blur_map)
        blurMap_blurnoblur[blur_map > 0] = 1
        accuracy_blur_no_blur = sess.run([accuracy_run], {output_map: np.expand_dims(blurMap_blurnoblur[:,:,np.newaxis],
                                                                                     axis=0),classification_map:
            np.expand_dims(gt_blur_no_blur, axis=0)})[0]
        accuracy_binary_list.append(accuracy_blur_no_blur)
        gt_test_image = np.squeeze(gt_test_image)
        blur_map = np.squeeze(blur_map)
        gt_blur_no_blur = np.squeeze(gt_blur_no_blur)
        all_gt_image_results.append(gt_test_image)
        all_image_results.append(blur_map)
        all_gt_binary_image_results.append(gt_blur_no_blur)
        all_binary_image_results.append(blurMap_blurnoblur)

        accuracy = accuracy_score(gt_test_image.flatten(),blur_map.flatten(),normalize=True)

        # calculate mean intersection of union
        miou = numpy_iou(gt_test_image,blur_map)
        miou_binary = numpy_iou(gt_blur_no_blur, blurMap_blurnoblur, 2)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        perclass_accuracy_conf_matrix = confusion_matrix(gt_test_image.flatten(),blur_map.flatten(),labels=[0,1,2,3,4],
                                                         normalize="true")

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(5):
            if (perclass_accuracy_conf_matrix[lab,:] == 0).all() and (perclass_accuracy_conf_matrix[:,lab] == 0).all():
                pass
            else:
                classesList[lab].append(perclass_accuracy[lab])

        perclass_accuracy_conf_matrix = confusion_matrix(gt_blur_no_blur.flatten(),blurMap_blurnoblur.flatten(),
                                                         labels=[0, 1], normalize="true")

        perclass_accuracy_binary = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(2):
            if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (
                    perclass_accuracy_conf_matrix[:, lab] == 0).all():
                pass
            else:
                classesListBinary[lab].append(perclass_accuracy_binary[lab])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(gt_test_image.flatten(),blur_map.flatten(), labels=[0,1,2,3,4],average='micro')

        f1scorebinary = f1_score(gt_blur_no_blur.flatten(),blurMap_blurnoblur.flatten(),labels=[0, 1],average='micro')
        f1score_binary_list.append(f1scorebinary)

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        miou_binary_list.append(miou_binary)
        f1score_list.append(f1score)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape)
        rgb_gt_map = np.zeros(test_image.shape)
        rgb_blur_map_blur_no_blur = np.zeros(test_image.shape)
        rgb_gt_map_blur_no_blur = np.zeros(test_image.shape)
        # blur no blur
        rgb_blur_map_blur_no_blur[blurMap_blurnoblur == 1] = [255, 255, 255]
        rgb_gt_map_blur_no_blur[gt_blur_no_blur == 1] = [255, 255, 255]
        # blue motion blur
        rgb_blur_map[blur_map == 1] = [255,0,0]
        rgb_gt_map[gt_test_image == 1] = [255,0,0]
        # green focus blur
        rgb_blur_map[blur_map == 2] = [0, 255, 0]
        rgb_gt_map[gt_test_image == 2] = [0, 255, 0]
        # red darkness blur
        rgb_blur_map[blur_map == 3] = [0, 0, 255]
        rgb_gt_map[gt_test_image == 3] = [0, 0, 255]
        # pink brightness blur
        rgb_blur_map[blur_map == 4] = [255, 192, 203]
        rgb_gt_map[gt_test_image == 4] = [255, 192, 203]
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

        log = "[*] Testing image name:" + image_name + " time: %4.4fs, Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
                                                       "Accuracy for Class 1 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (time.time() - start_time, accuracy_blur_no_blur, perclass_accuracy_binary[0],
                 perclass_accuracy_binary[1], miou_binary, f1scorebinary)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics_bnb.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image_name, str(np.round(accuracy_blur_no_blur, 8)),
                       str(np.round(perclass_accuracy_binary[0], 8)),
                       str(np.round(perclass_accuracy_binary[1], 8)), str(np.round(miou_binary, 8)),
                       str(np.round(f1scorebinary, 8))]
        with open(save_dir_sample + "/testing_metrics_bnb.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(string_list)

        if ".jpg" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)
        if ".JPG" in image_name:
            image_name.replace(".JPG", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".JPG", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".JPG", ".png"), rgb_blur_map_blur_no_blur)
            cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".JPG", ".png"),
                        rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".JPG", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".JPG", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".JPG", ".png"), rgb_o3)
        if ".PNG" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),
                        rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)
        if ".png" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),
                        rgb_gt_map_blur_no_blur)
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
    log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f " \
          "Max IoU: %.8f Variance: %.8f Max F1_score: %.8f\n" % (np.max(np.array(accuracy_binary_list)),
                                                   np.max(np.array(classesListBinary[0]), axis=0),
                                                   np.max(np.array(classesListBinary[1]), axis=0),
                                                   np.max(np.array(miou_binary_list)),
                                                   np.var(np.asarray(accuracy_binary_list)),
                                                   np.max(np.array(f1score_binary_list)))
    log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
            "Mean IoU: %.8f Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_binary_list)),
                                                      np.mean(np.array(classesListBinary[0])),
                                                      np.mean(np.array(classesListBinary[1])),
                                                      np.mean(np.array(miou_binary_list)),
                                                      np.mean(np.array(f1score_binary_list)))
    # only way to write to log file while running
    with open(save_dir_sample + "/testing_metrics_bnb.log", "a") as f:
        # perform file operations
        f.write(log)
        f.write(log2)

    classesListBinary,classesList = None,None

    # plot confusion matrix for all classification labels
    plt.rc('font', size=20)  # controls default text size
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=30)  # fontsize of the legend
    plt.rcParams['figure.figsize'] = 10, 10
    plt.clf()
    final_confmatrix = confusion_matrix(np.array(all_gt_image_results).flatten(), np.array(all_image_results).flatten(),
                                        labels=[0, 1, 2, 3, 4], normalize="true")
    all_gt_image_results = None
    all_image_results = None
    np.save(save_dir_sample + '/muri_confusion_matrix_results.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix,3)
    plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Motion', 'Focus','Darkness','Brightness']
    plt.title('MURI Weights - Test Data Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    thresh = final_confmatrix.max() / 2.
    for i in range(final_confmatrix.shape[0]):
        for j in range(final_confmatrix.shape[1]):
            plt.text(j, i, format(final_confmatrix[i, j]),
                     ha="center", va="center",
                     color="white" if final_confmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(save_dir_sample + '/muri_conf_matrix_all_labels.png')
    plt.show()

    # https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
    plt.rcParams['figure.figsize'] = 8, 5
    plt.clf()
    final_confmatrix = confusion_matrix(np.array(all_gt_binary_image_results).flatten(),
                                        np.array(all_binary_image_results).flatten(), labels=[0, 1], normalize="true")
    all_gt_binary_image_results = None
    all_binary_image_results = None
    np.save(save_dir_sample + '/muri_confusion_binary_matrix_results.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix, 3)
    plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Blur']
    plt.title('MURI Weights - Test Data Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    thresh = final_confmatrix.max() / 2.
    for i in range(final_confmatrix.shape[0]):
        for j in range(final_confmatrix.shape[1]):
            plt.text(j, i, format(final_confmatrix[i, j]),
                     ha="center", va="center",
                     color="white" if final_confmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(save_dir_sample + '/muri_conf_matrix_binary_labels.png')
    plt.show()

# main test function
def test_sensitivity_with_muri_dataset():
    print("MURI Testing")

    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample + '/gt')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.SENSITVITY.muri_blur_path, regx='/*.(png|PNG)',
                                                        printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.SENSITVITY.muri_gt_path, regx='/*.(png|PNG)',
                                                        printable=False))

    ###Load Testing Data ####
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.SENSITVITY.muri_blur_path, n_threads=100, mode='RGB')
    test_mask_imgs = read_all_imgs(test_mask_img_list, path=config.SENSITVITY.muri_gt_path, n_threads=100,
                                   mode='RGB2GRAY2')

    test_classification_mask = []
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

    h = test_blur_imgs[0].shape[0]
    w = test_blur_imgs[0].shape[1]

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [1, h, w, 3], name='input_patches')
    classification_map = tf.compat.v1.placeholder('int64', [1, h, w, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3, m1B, m2B, m3B, m4B = Updated_Decoder_Network_classification(input, n, f0, f0_1, f1_2,
                                                                                            f2_3, reuse=False,
                                                                                            scope=scope2)

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
    output = tf.nn.softmax(net_regression.outputs)

    ### DEFINE LOSS ###
    accuracy_run = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(output_map, classification_map))),
                    dtype=tf.float32)*(1/(h*w))

    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Load checkpoint
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    file_name = './model/SA_net_{}.ckpt'.format(tl.global_flag['mode'])
    reader = py_checkpoint_reader.NewCheckpointReader(file_name)
    state_dict = {v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()}
    get_weights_checkpoint(sess, net_regression, state_dict)

    accuracy_list = []
    miou_list = []
    miou_binary_list = []
    f1score_list = []
    classesListBinary = [[],[],[],[],[]]
    classesList = [[], [], [], [], []]
    accuracy_binary_list = []
    f1score_binary_list = []
    all_gt_binary_image_results = []
    all_gt_image_results = []
    all_binary_image_results = []
    all_image_results = []

    # initalize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1',
                         'Accuracy for Class 2','Accuracy for Class 3','Accuracy for Class 4','mIOU','f1_score'])

    with open(save_dir_sample + "/testing_metrics_bnb.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1'])

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
        gt_blur_no_blur = np.copy(gt_test_image)
        gt_blur_no_blur[gt_test_image > 0] = 1
        # uncertain labeling
        if tl.global_flag['uncertainty_label']:
            blur_map = np.zeros((h, w))
            blurMap = np.squeeze(sess.run([output], {net_regression.inputs: np.expand_dims((test_image), axis=0)}))
            # numpy array with labels
            blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:, :] >=
                                                                        threshold,axis=2) == 1],axis=1)
            # uncertainty labeling
            blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) != 1] = 5
        else:
            blurMap = sess.run([output_map], {net_regression.inputs: np.expand_dims(test_image, axis=0)})[0]
            blur_map = np.squeeze(blurMap)
        blurMap_blurnoblur = np.copy(blur_map)
        blurMap_blurnoblur[blur_map > 0] = 1
        accuracy_blur_no_blur = sess.run([accuracy_run], {output_map: np.expand_dims(blurMap_blurnoblur[:,:,np.newaxis],
                                                                                     axis=0),classification_map:
            np.expand_dims(gt_blur_no_blur, axis=0)})[0]
        accuracy_binary_list.append(accuracy_blur_no_blur)
        gt_test_image = np.squeeze(gt_test_image)
        blur_map = np.squeeze(blur_map)
        gt_blur_no_blur = np.squeeze(gt_blur_no_blur)
        all_gt_binary_image_results.append(gt_blur_no_blur)
        all_image_results.append(blur_map)
        all_gt_image_results.append(gt_test_image)
        all_binary_image_results.append(blurMap_blurnoblur)

        accuracy = accuracy_score(gt_test_image.flatten(),blur_map.flatten(),normalize=True)

        # calculate mean intersection of union
        miou = numpy_iou(gt_test_image,blur_map)
        miou_binary = numpy_iou(gt_blur_no_blur, blurMap_blurnoblur, 2)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        perclass_accuracy_conf_matrix = confusion_matrix(gt_test_image.flatten(),blur_map.flatten(),labels=[0,1,2,3,4],
                                                         normalize="true")

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(5):
            if (perclass_accuracy_conf_matrix[lab,:] == 0).all() and (perclass_accuracy_conf_matrix[:,lab] == 0).all():
                pass
            else:
                classesList[lab].append(perclass_accuracy[lab])

        perclass_accuracy_conf_matrix = confusion_matrix(gt_blur_no_blur.flatten(),blurMap_blurnoblur.flatten(),
                                                         labels=[0, 1], normalize="true")

        perclass_accuracy_binary = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(2):
            if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (
                    perclass_accuracy_conf_matrix[:, lab] == 0).all():
                pass
            else:
                classesListBinary[lab].append(perclass_accuracy_binary[lab])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(gt_test_image.flatten(),blur_map.flatten(), labels=[0,1,2,3,4],average='micro')

        f1scorebinary = f1_score(gt_blur_no_blur.flatten(),blurMap_blurnoblur.flatten(),labels=[0, 1],average='micro')
        f1score_binary_list.append(f1scorebinary)

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        miou_binary_list.append(miou_binary)
        f1score_list.append(f1score)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape)
        rgb_gt_map = np.zeros(test_image.shape)
        rgb_blur_map_blur_no_blur = np.zeros(test_image.shape)
        rgb_gt_map_blur_no_blur = np.zeros(test_image.shape)
        # blur no blur
        rgb_blur_map_blur_no_blur[blurMap_blurnoblur == 1] = [255, 255, 255]
        rgb_gt_map_blur_no_blur[gt_blur_no_blur == 1] = [255, 255, 255]
        # blue motion blur
        rgb_blur_map[blur_map == 1] = [255,0,0]
        rgb_gt_map[gt_test_image == 1] = [255,0,0]
        # green focus blur
        rgb_blur_map[blur_map == 2] = [0, 255, 0]
        rgb_gt_map[gt_test_image == 2] = [0, 255, 0]
        # red darkness blur
        rgb_blur_map[blur_map == 3] = [0, 0, 255]
        rgb_gt_map[gt_test_image == 3] = [0, 0, 255]
        # pink brightness blur
        rgb_blur_map[blur_map == 4] = [255, 192, 203]
        rgb_gt_map[gt_test_image == 4] = [255, 192, 203]
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

        log = "[*] Testing image name:" + image_name + " time: %4.4fs, Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
                                                       "Accuracy for Class 1 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (time.time() - start_time, accuracy_blur_no_blur, perclass_accuracy_binary[0],
                 perclass_accuracy_binary[1], miou_binary, f1scorebinary)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics_bnb.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image_name, str(np.round(accuracy_blur_no_blur, 8)),
                       str(np.round(perclass_accuracy_binary[0], 8)),
                       str(np.round(perclass_accuracy_binary[1], 8)), str(np.round(miou_binary, 8)),
                       str(np.round(f1scorebinary, 8))]
        with open(save_dir_sample + "/testing_metrics_bnb.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(string_list)

        if ".jpg" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)
        if ".JPG" in image_name:
            image_name.replace(".JPG", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".JPG", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".JPG", ".png"), rgb_blur_map_blur_no_blur)
            cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".JPG", ".png"),
                        rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".JPG", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".JPG", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".JPG", ".png"), rgb_o3)
        if ".PNG" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),
                        rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)
        if ".png" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),
                        rgb_gt_map_blur_no_blur)
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
    log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f " \
          "Max IoU: %.8f Variance: %.8f Max F1_score: %.8f\n" % (np.max(np.array(accuracy_binary_list)),
                                                   np.max(np.array(classesListBinary[0]), axis=0),
                                                   np.max(np.array(classesListBinary[1]), axis=0),
                                                   np.max(np.array(miou_binary_list)),
                                                   np.var(np.asarray(accuracy_binary_list)),
                                                   np.max(np.array(f1score_binary_list)))
    log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
            "Mean IoU: %.8f Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_binary_list)),
                                                      np.mean(np.array(classesListBinary[0])),
                                                      np.mean(np.array(classesListBinary[1])),
                                                      np.mean(np.array(miou_binary_list)),
                                                      np.mean(np.array(f1score_binary_list)))
    # only way to write to log file while running
    with open(save_dir_sample + "/testing_metrics_bnb.log", "a") as f:
        # perform file operations
        f.write(log)
        f.write(log2)

    classesListBinary,classesList = None,None

    # plot confusion matrix for all classification labels
    plt.rc('font', size=20)  # controls default text size
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=30)  # fontsize of the legend
    # https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
    plt.rcParams['figure.figsize'] = 10, 10
    plt.clf()
    final_confmatrix = confusion_matrix(np.array(all_gt_image_results).flatten(), np.array(all_image_results).flatten(),
                                        labels=[0, 1, 2, 3, 4], normalize="true")
    all_gt_image_results = None
    all_image_results = None
    np.save(save_dir_sample + '/muri_confusion_matrix_results.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix,3)
    plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Motion', 'Focus','Darkness','Brightness']
    plt.title('MURI Weights - Test Data Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    thresh = final_confmatrix.max() / 2.
    for i in range(final_confmatrix.shape[0]):
        for j in range(final_confmatrix.shape[1]):
            plt.text(j, i, format(final_confmatrix[i, j]),
                     ha="center", va="center",
                     color="white" if final_confmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(save_dir_sample + '/muri_conf_matrix_all_labels.png')
    plt.show()

    plt.rcParams['figure.figsize'] = 8, 5
    plt.clf()
    final_confmatrix = confusion_matrix(np.array(all_gt_binary_image_results).flatten(),
                                        np.array(all_binary_image_results).flatten(), labels=[0, 1], normalize="true")
    all_gt_binary_image_results = None
    all_binary_image_results = None
    np.save(save_dir_sample + '/muri_confusion_binary_matrix_results.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix, 3)
    plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Blur']
    plt.title('MURI Weights - Test Data Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    thresh = final_confmatrix.max() / 2.
    for i in range(final_confmatrix.shape[0]):
        for j in range(final_confmatrix.shape[1]):
            plt.text(j, i, format(final_confmatrix[i, j]),
                     ha="center", va="center",
                     color="white" if final_confmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(save_dir_sample + '/muri_conf_matrix_binary_labels.png')
    plt.show()

# main test function
def test_with_real_bd_dataset():
    print("Real BD Testing")

    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    #tl.files.exists_or_mkdir(save_dir_sample + '/gt')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary')
    #tl.files.exists_or_mkdir(save_dir_sample + '/binary/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.real_blur_path, regx='/*.(jpg|JPG)',
                                                        printable=False))
    # test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_gt_path, regx='/*.(png|PNG)',
    #                                                     printable=False))

    ###Load Testing Data ####
    # test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.real_blur_path, n_threads=100, mode='RGB')

    # # Load checkpoint
    # # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    file_name = './model/SA_net_{}.ckpt'.format(tl.global_flag['mode'])
    reader = py_checkpoint_reader.NewCheckpointReader(file_name)
    state_dict = {v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()}
    # get_weights_checkpoint(sess, net_regression, state_dict)

    for i in range(len(test_blur_img_list)):
        #test_image = test_blur_imgs[i]
        #gt_test_image = test_mask_imgs[i]
        image_name = test_blur_img_list[i]
        image_file_location = os.path.join(config.TEST.real_blur_path, image_name)
        test_image = imageio.imread(image_file_location)

        sharp_image = np.asarray(test_image, dtype="float32")

        if len(sharp_image.shape) < 3:
            sharp_image = np.expand_dims(np.asarray(sharp_image), 3)
            sharp_image = np.concatenate([sharp_image, sharp_image, sharp_image], axis=2)

        if sharp_image.shape[2] == 4:
            print(sharp_image.shape)
            sharp_image = np.expand_dims(np.asarray(sharp_image), 3)

            print(sharp_image.shape)
            sharp_image = np.concatenate((sharp_image[:, :, 0], sharp_image[:, :, 1], sharp_image[:, :, 2]), axis=2)

        print(sharp_image.shape)

        image_h, image_w = sharp_image.shape[0:2]
        print(image_h, image_w)

        test_image = sharp_image[0: image_h - (image_h % 16), 0: 0 + image_w - (image_w % 16), :]
        red = test_image[:, :, 0]
        green = test_image[:, :, 1]
        blue = test_image[:, :, 2]
        test_image = np.zeros(test_image.shape)
        test_image[:, :, 0] = blue - VGG_MEAN[0]
        test_image[:, :, 1] = green - VGG_MEAN[1]
        test_image[:, :, 2] = red - VGG_MEAN[2]

        # Model
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=run_session,args=(test_image, state_dict, return_dict))
        p.start()
        p.join()
        # run_session(bgr,test_gt_image,return_dict)
        blur_map,blur_map_binary = return_dict.values()
        blur_map = np.squeeze(blur_map)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape)
        #rgb_gt_map = np.zeros(test_image.shape)
        rgb_blur_map_blur_no_blur = np.zeros(test_image.shape)
        #rgb_gt_map_blur_no_blur = np.zeros(test_image.shape)
        # blur no blur
        rgb_blur_map_blur_no_blur[blur_map_binary == 1] = [255, 255, 255]
        #rgb_gt_map_blur_no_blur[gt_blur_no_blur == 1] = [255, 255, 255]
        # blue motion blur
        rgb_blur_map[blur_map == 1] = [255,0,0]
        #rgb_gt_map[gt_test_image == 1] = [255,0,0]
        # green focus blur
        rgb_blur_map[blur_map == 2] = [0, 255, 0]
        #rgb_gt_map[gt_test_image == 2] = [0, 255, 0]
        # red darkness blur
        rgb_blur_map[blur_map == 3] = [0, 0, 255]
        #rgb_gt_map[gt_test_image == 3] = [0, 0, 255]
        # pink brightness blur
        rgb_blur_map[blur_map == 4] = [255, 192, 203]
        #rgb_gt_map[gt_test_image == 4] = [255, 192, 203]
        # yellow unknown blur
        rgb_blur_map[blur_map == 5] = [0, 255, 255]

        if ".jpg" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            #cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            #cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)
        if ".JPG" in image_name:
            image_name.replace(".JPG", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".JPG", ".png"), rgb_blur_map)
            #cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".JPG", ".png"), rgb_blur_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".JPG", ".png"),
            #             rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".JPG", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".JPG", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".JPG", ".png"), rgb_o3)
        if ".PNG" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            #cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),
            #             rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)
        if ".png" in image_name:
            image_name.replace(".jpg", ".png")
            cv2.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            #cv2.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            cv2.imwrite(save_dir_sample + '/binary/' + image_name.replace(".jpg", ".png"), rgb_blur_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/binary/gt/gt_' + image_name.replace(".jpg", ".png"),
            #             rgb_gt_map_blur_no_blur)
            # cv2.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_o1)
            # cv2.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_o2)
            # cv2.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_o3)

# multiprocessing run session
def run_session(image, state_dict, return_dict):
    # Model
    patches_blurred = tf.compat.v1.placeholder('float32', [1, image.shape[0], image.shape[1], 3],
                                               name='input_patches')
    with tf.compat.v1.variable_scope('Unified') as scope:
        with tf.compat.v1.variable_scope('VGG') as scope3:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope3)
        with tf.compat.v1.variable_scope('UNet') as scope1:
            net_regression, m1, m2, m3 = Updated_Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3,
                                                                        reuse=False, scope=scope1)

    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    #configTf.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)

    # Load checkpoint
    #get_weights(sess, net_regression)
    get_weights_checkpoint(sess, net_regression, state_dict)
    print("loaded all the weights")

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
    # output_map_1 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m1.outputs), axis=3), axis=3)
    # output_map_2 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m2.outputs), axis=3), axis=3)
    # output_map_3 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m3.outputs), axis=3), axis=3)

    blur_map0 = sess.run([output_map],{net_regression.inputs: np.expand_dims(image, axis=0)})[0]
    # compare binary map
    blur_map_binary = np.copy(blur_map0)
    blur_map_binary[blur_map0 > 0] = 1
    blur_map_binary = np.squeeze(blur_map_binary)

    sess.close()
    return_dict['blur_map'] = blur_map0
    # return_dict['blur_map_1'] = outmap1
    # return_dict['blur_map_2'] = outmap2
    # return_dict['blur_map_3'] = outmap3
    return_dict['blur_map_binary'] = blur_map_binary

# our test
def test():
    print("Blurmap Testing")

    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample + '/gt')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary_result/')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary_result/gt')

    test_blur_img_list = sorted(
        tl.files.load_file_list(path=config.TEST.blur_path, regx='/*.(png|PNG)', printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.gt_path, regx='/*.(png|PNG)', printable=False))

    ###Load Testing Data ####
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.blur_path + '/', n_threads=100, mode='RGB')
    test_mask_imgs = read_all_imgs(test_mask_img_list, path=config.TEST.gt_path + '/', n_threads=100, mode='RGB2GRAY2')

    test_classification_mask = []
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
    h = config.TEST.height
    w = config.TEST.width

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [1, h, w, 3], name='input_patches')
    labels = tf.compat.v1.placeholder('int64', [1, h, w, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3 = Updated_Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3,
                                                                                                    reuse=False,
                                                                                                    scope=scope2)

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
    output = tf.nn.softmax(net_regression.outputs)
    # layer output definition
    output_map1 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m1.outputs),axis=3),axis=3)
    output_map2 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m2.outputs),axis=3),axis=3)
    output_map3 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m3.outputs),axis=3),axis=3)

    ### DEFINE LOSS ###
    loss1 = tf.cast(tf.math.reduce_sum(1 - tf.math.abs(tf.math.subtract(output_map, labels))),
                    dtype=tf.float32) * (1 / (h * w))

    # Load checkpoint
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)
    # read check point weights
    reader = py_checkpoint_reader.NewCheckpointReader('./model/SA_net_{}.ckpt'.format(tl.global_flag['mode']))
    state_dict = {v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()}
    # save weights to the model
    get_weights_checkpoint(sess, net_regression, state_dict)

    net_regression.test()
    m1.test()
    m2.test()
    m3.test()

    accuracy_list = []
    miou_list = []
    f1score_list = []
    classesList = [[], [], [], [], []]
    accuracy_list_binary = []
    miou_list_binary = []
    f1score_binary_list = []
    classesListBinary = [[], []]
    # initialize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1', 'Accuracy for Class 2',
             'Accuracy for Class 3', 'Accuracy for Class 4', 'mIOU', 'f1_score'])

    with open(save_dir_sample + "/testing_metrics_binary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1', 'mIOU',
                         'f1_score'])
    f = None
    writer = None

    all_img_results = []
    all_gt_img_results = []
    all_img_binary_results = []
    all_gt_binary_results = []

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

        # need to have blur-no-blur comparison
        test_gt_image_blur_no_blur = np.copy(gt_test_image)
        # only 1 label
        test_gt_image_blur_no_blur[gt_test_image > 0] = 1

        # Model
        start_time = time.time()
        # uncertain labeling
        if tl.global_flag['uncertainty_label']:
            blur_map = np.zeros((h, w))
            blurMap, o1, o2, o3 = sess.run([output, output_map1, output_map2, output_map3],
                                                      {net_regression.inputs: np.expand_dims(test_image, axis=0)})
            blurMap = np.squeeze(blurMap)
            # numpy array with labels
            blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:, :] >=
                                                                                                 threshold,
                                                                                                 axis=2) == 1], axis=1)
            # uncertainty labeling
            blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) != 1] = 5
            blur_map = blur_map[np.newaxis,:,:,np.newaxis]
        else:
            blur_map, o1, o2, o3 = sess.run([output_map, output_map1, output_map2, output_map3],
                                            {net_regression.inputs: np.expand_dims(test_image, axis=0)})
            #blur_map = np.squeeze(blurMap)

        # blur_map = sess.run([output_map], {net_regression.inputs: np.expand_dims(test_image, axis=0)})[0]
        #blur_map,o1,o2,o3 = sess.run([output_map,output_map1,output_map2,output_map3], {net_regression.inputs: np.expand_dims(test_image, axis=0)})

        accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(), np.squeeze(blur_map).flatten(), normalize=True)
        # compare binary map
        blur_map_binary = np.copy(blur_map)
        blur_map_binary[blur_map > 0] = 1

        accuracy_binary = sess.run([loss1], {output_map: blur_map_binary, labels: np.expand_dims(
            test_gt_image_blur_no_blur, axis=0)})[0]

        blur_map = np.squeeze(blur_map)
        blur_map_binary = np.squeeze(blur_map_binary)
        test_gt_image_blur_no_blur = np.squeeze(test_gt_image_blur_no_blur)
        gt_test_image = np.squeeze(gt_test_image)
        # output layers
        o1 = np.squeeze(o1)
        o2 = np.squeeze(o2)
        o3 = np.squeeze(o3)

        # calculate mean intersection of union
        miou = numpy_iou(gt_test_image, blur_map, 5)
        miou_binary = numpy_iou(test_gt_image_blur_no_blur, blur_map_binary, 2)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        perclass_accuracy_conf_matrix = confusion_matrix(gt_test_image.flatten(), blur_map.flatten(),
                                                         labels=[0, 1, 2, 3, 4],
                                                         normalize="true")
        all_img_results.append(blur_map)
        all_gt_img_results.append(gt_test_image)
        all_img_binary_results.append(blur_map_binary)
        all_gt_binary_results.append(test_gt_image_blur_no_blur)

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(5):
            if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (
                    perclass_accuracy_conf_matrix[:, lab] == 0).all():
                pass
            else:
                classesList[lab].append(perclass_accuracy[lab])

        perclass_accuracy_conf_matrix = confusion_matrix(test_gt_image_blur_no_blur.flatten(),
                                                         blur_map_binary.flatten(),
                                                         labels=[0, 1], normalize="true")

        perclass_accuracy_binary = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(2):
            if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (
                    perclass_accuracy_conf_matrix[:, lab] == 0).all():
                pass
            else:
                classesListBinary[lab].append(perclass_accuracy_binary[lab])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(gt_test_image.flatten(), blur_map.flatten(), labels=[0, 1, 2, 3, 4], average='micro')
        f1score_binary = f1_score(test_gt_image_blur_no_blur.flatten(), blur_map_binary.flatten(), labels=[0, 1],
                                  average='micro')

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        f1score_list.append(f1score)
        accuracy_list_binary.append(accuracy_binary)
        miou_list_binary.append(miou_binary)
        f1score_binary_list.append(f1score_binary)

        blur_map = np.squeeze(blur_map)
        gt_map = np.squeeze(gt_test_image)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape).astype(np.uint8)
        rgb_blur_map_1 = np.zeros((o1.shape[0],o1.shape[1],3)).astype(np.uint8)
        rgb_blur_map_2 = np.zeros((o2.shape[0], o2.shape[1], 3)).astype(np.uint8)
        rgb_blur_map_3 = np.zeros((o3.shape[0], o3.shape[1], 3)).astype(np.uint8)
        rgb_gt_map = np.zeros(test_image.shape).astype(np.uint8)
        rgb_gt_map_1 = np.zeros((o1.shape[0], o1.shape[1], 3)).astype(np.uint8)
        rgb_gt_map_2 = np.zeros((o2.shape[0], o2.shape[1], 3)).astype(np.uint8)
        rgb_gt_map_3 = np.zeros((o3.shape[0], o3.shape[1], 3)).astype(np.uint8)
        gt_o1 = np.squeeze(cv2.resize(gt_test_image, (o1.shape[1], o1.shape[0]), interpolation=cv2.INTER_NEAREST))
        gt_o2 = np.squeeze(cv2.resize(gt_test_image, (o2.shape[1], o2.shape[0]), interpolation=cv2.INTER_NEAREST))
        gt_o3 = np.squeeze(cv2.resize(gt_test_image, (o3.shape[1], o3.shape[0]), interpolation=cv2.INTER_NEAREST))

        # red motion blur
        rgb_blur_map[blur_map == 1] = [255, 0, 0]
        rgb_blur_map_1[o1 == 1] = [255, 0, 0]
        rgb_blur_map_2[o2 == 1] = [255, 0, 0]
        rgb_blur_map_3[o3 == 1] = [255, 0, 0]
        rgb_gt_map[gt_map == 1] = [255, 0, 0]
        rgb_gt_map_1[gt_o1 == 1] = [255, 0, 0]
        rgb_gt_map_2[gt_o2 == 1] = [255, 0, 0]
        rgb_gt_map_3[gt_o3 == 1] = [255, 0, 0]
        # green focus blur
        rgb_blur_map[blur_map == 2] = [0, 255, 0]
        rgb_blur_map_1[o1 == 2] = [0, 255, 0]
        rgb_blur_map_2[o2 == 2] = [0, 255, 0]
        rgb_blur_map_3[o3 == 2] = [0, 255, 0]
        rgb_gt_map[gt_map == 2] = [0, 255, 0]
        rgb_gt_map_1[gt_o1 == 2] = [0, 255, 0]
        rgb_gt_map_2[gt_o2 == 2] = [0, 255, 0]
        rgb_gt_map_3[gt_o3 == 2] = [0, 255, 0]
        # blue darkness blur
        rgb_blur_map[blur_map == 3] = [0, 0, 255]
        rgb_blur_map_1[o1 == 3] = [0, 0, 255]
        rgb_blur_map_2[o2 == 3] = [0, 0, 255]
        rgb_blur_map_3[o3 == 3] = [0, 0, 255]
        rgb_gt_map[gt_map == 3] = [0, 0, 255]
        rgb_gt_map_1[gt_o1 == 3] = [0, 0, 255]
        rgb_gt_map_2[gt_o2 == 3] = [0, 0, 255]
        rgb_gt_map_3[gt_o3 == 3] = [0, 0, 255]
        # pink brightness blur
        rgb_blur_map[blur_map == 4] = [255, 192, 203]
        rgb_blur_map_1[o1 == 4] = [255, 192, 203]
        rgb_blur_map_2[o2 == 4] = [255, 192, 203]
        rgb_blur_map_3[o3 == 4] = [255, 192, 203]
        rgb_gt_map[gt_map == 4] = [255, 192, 203]
        rgb_gt_map_1[gt_o1 == 4] = [255, 192, 203]
        rgb_gt_map_2[gt_o2 == 4] = [255, 192, 203]
        rgb_gt_map_3[gt_o3 == 4] = [255, 192, 203]
        # yellow unknown blur
        rgb_blur_map[blur_map == 5] = [0, 255, 255]

        log = "[*] Testing image name:" + image_name + "time: %4.4fs, Overall Accuracy: %.8f Accuracy for Class 0 " \
                                                       "%.8f " \
                                                       "Accuracy for Class 1 %.8f Accuracy for Class 2 %.8f Accuracy " \
                                                       "for Class 3 %.8f " \
                                                       "Accuracy for Class 4 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (time.time() - start_time, accuracy, perclass_accuracy[0], perclass_accuracy[1], perclass_accuracy[2],
                 perclass_accuracy[3], perclass_accuracy[4], miou, f1score)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image_name, str(np.round(accuracy, 8)), str(np.round(perclass_accuracy[0], 8)),
                       str(np.round(perclass_accuracy[1], 8)), str(np.round(perclass_accuracy[2], 8)),
                       str(np.round(perclass_accuracy[3], 8)), str(np.round(perclass_accuracy[4], 8)),
                       str(np.round(miou, 8)),
                       str(np.round(f1score, 8))]
        with open(save_dir_sample + "/testing_metrics.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(string_list)

        log = "[*] Testing image name:" + image_name + " Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
                                                       "Accuracy for Class 1 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (accuracy_binary, perclass_accuracy_binary[0], perclass_accuracy_binary[1], miou_binary, f1score_binary)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics_binary.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image_name, str(np.round(accuracy_binary, 8)), str(np.round(perclass_accuracy_binary[0], 8)),
                       str(np.round(perclass_accuracy_binary[1], 8)), str(np.round(miou, 8)),
                       str(np.round(f1score, 8))]
        with open(save_dir_sample + "/testing_metrics_binary.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(string_list)

        if ".jpg" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".JPG" in image_name:
            image_name.replace(".JPG", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".JPG", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map)
            imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_1)
            imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_1)
            imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_2)
            imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_2)
            imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_3)
            imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".JPG", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".JPG", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".PNG" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".png" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)

    sess.close()
    log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f " \
          "Max Accuracy Class 2: %.8f Max Accuracy Class 3: %.8f Max Accuracy Class 4: %.8f Max IoU: %.8f " \
          "Variance: %.8f Max F1_score: %.8f\n" % (np.max(np.array(accuracy_list)),
                                                   np.max(np.array(classesList[0]), axis=0),
                                                   np.max(np.array(classesList[1]), axis=0),
                                                   np.max(np.array(classesList[2]), axis=0),
                                                   np.max(np.array(classesList[3]), axis=0),
                                                   np.max(np.array(classesList[4]), axis=0),
                                                   np.max(np.array(miou_list)), np.var(np.asarray(accuracy_list)),
                                                   np.max(np.array(f1score_list)))
    log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
           "Mean Accuracy Class 2: %.8f Mean Accuracy Class 3: %.8f Mean Accuracy Class 4: %.8f Mean IoU: %.8f " \
           "Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_list)), np.mean(np.array(classesList[0])),
                                      np.mean(np.array(classesList[1])), np.mean(np.array(classesList[2])),
                                      np.mean(np.array(classesList[3])),
                                      np.mean(np.array(classesList[4])), np.mean(np.array(miou_list)),
                                      np.mean(np.array(f1score_list)))
    # only way to write to log file while running
    with open(save_dir_sample + "/testing_metrics.log", "a") as f:
        # perform file operations
        f.write(log)
        f.write(log2)

    log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f Max IoU: %.8f " \
          "Variance: %.8f Max F1_score: %.8f\n" % (
              np.max(np.array(accuracy_list_binary)),
              np.max(np.array(classesListBinary[0]), axis=0), np.max(np.array(classesListBinary[1]), axis=0),
              np.max(np.array(miou_list_binary)), np.var(np.asarray(accuracy_list_binary)),
              np.max(np.array(f1score_binary_list)))
    log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
           "Mean IoU: %.8f Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_list_binary)),
                                                     np.mean(np.array(classesListBinary[0])),
                                                     np.mean(np.array(classesListBinary[1])),
                                                     np.mean(np.array(miou_list_binary)),
                                                     np.mean(np.array(f1score_binary_list)))
    # only way to write to log file while running
    with open(save_dir_sample + "/testing_metrics_binary.log", "a") as f:
        # perform file operations
        f.write(log)
        f.write(log2)

    plt.rc('font', size=30)  # controls default text size
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=30)  # fontsize of the legend

    plt.clf()
    plt.figure(figsize=(12, 10))
    final_confmatrix = confusion_matrix(np.array(all_gt_img_results).flatten(), np.array(all_img_results).flatten(),
                                        labels=[0, 1, 2, 3, 4], normalize="true")
    np.save(save_dir_sample +'/all_labels_results_conf_matrix.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix, 2)
    cax = plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Motion', 'Focus', 'Darkness', 'Brightness']
    plt.title('Our Weights - Test Data Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    thresh = final_confmatrix.max() / 2.
    for i in range(final_confmatrix.shape[0]):
        for j in range(final_confmatrix.shape[1]):
            plt.text(j, i, format(final_confmatrix[i, j]),
                     ha="center", va="center",
                     color="white" if final_confmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    cbar = plt.colorbar(cax, ticks=[final_confmatrix.min(), final_confmatrix.max()])
    cbar.ax.set_yticklabels(['0.0', ' 1.0'])
    plt.savefig(save_dir_sample +'/conf_matrix_all_labels.png')
    plt.show()

    plt.clf()
    plt.figure(figsize=(8, 7))
    final_confmatrix = confusion_matrix(np.array(all_gt_binary_results).flatten(),
                                        np.array(all_img_binary_results).flatten(), labels=[0, 1], normalize="true")
    np.save(save_dir_sample +'/all_binary_results_conf_matrix.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix, 2)
    cax = plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Blur']
    plt.title('Our Weights - Test Data Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    if final_confmatrix.min() > .3:
        thresh = final_confmatrix.min()
    else:
        thresh = final_confmatrix.max() / 2.
    for i in range(final_confmatrix.shape[0]):
        for j in range(final_confmatrix.shape[1]):
            plt.text(j, i, format(final_confmatrix[i, j]),
                     ha="center", va="center",
                     color="white" if final_confmatrix[i, j] > thresh else "black")
    plt.tight_layout()
    cbar = plt.colorbar(cax, ticks=[final_confmatrix.min(), final_confmatrix.max()])
    cbar.ax.set_yticklabels(['0.0', ' 1.0'])
    plt.savefig(save_dir_sample +'/conf_matrix_binary.png')
    plt.show()