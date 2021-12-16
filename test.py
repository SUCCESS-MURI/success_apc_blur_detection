# coding=utf-8
import csv

import imageio
import tensorflow as tf
from matplotlib import pyplot as plt

tf.compat.v1.disable_eager_execution()
import tensorlayer as tl
import numpy as np
import math
from tensorflow.python.training import py_checkpoint_reader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from config import config
from utils import read_all_imgs, get_imgs_RGB_fn, get_imgs_RGBGRAY_2_fn
from model import Decoder_Network_classification, VGG19_pretrained, Decoder_Network_classification_3_labels
import matplotlib
import datetime
import time
import cv2

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
    dict_weights_trained = np.load('inital_final_model.npy',allow_pickle=True)[()]
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

# main test function
def test_with_muri_dataset():
    print("MURI Testing")

    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample+'/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_blur_path, regx='/*.(png|PNG)', printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_gt_path, regx='/*.(png|PNG)', printable=False))

    ###Load Testing Data ####
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.muri_blur_path, n_threads=100, mode='RGB')
    test_mask_imgs = read_all_imgs(test_mask_img_list, path=config.TEST.muri_gt_path, n_threads=100, mode='RGB2GRAY2')

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

    h = test_blur_imgs[0].shape[0]
    w = test_blur_imgs[0].shape[1]

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
    accuracy_run = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(output_map, classification_map))),
                    dtype=tf.float32)*(1/(h*w))
    #mean_iou, update_op = tf.compat.v1.metrics.mean_iou(labels=classification_map,predictions=predictions,num_classes=5)

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
        blurMap, accuracy = sess.run([output, accuracy_run],
                                                {net_regression.inputs: np.expand_dims(test_image, axis=0),
                                                 classification_map: np.expand_dims(gt_test_image, axis=0)})
        blurMap = np.squeeze(blurMap)
        gt_test_image = np.squeeze(gt_test_image)
        if tl.global_flag['uncertainty_label']:
            blur_map = np.zeros((h, w))
            blur_map[np.sum(blurMap[:,:] >= .4,axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:,:] >= .4,axis=2) == 1],
                                                                         axis=1)
            # uncertainty labeling
            blur_map[np.sum(blurMap[:, :] >= .4, axis=2) != 1] = 5
        else:
            blur_map = np.argmax(blurMap,axis=2)

        #np.save(save_dir_sample + '/raw_' + image_name.replace(".png", ".npy"), np.squeeze(blur_map))
        #accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(),np.squeeze(blur_map).flatten(),normalize=True)

        # calculate mean intersection of union
        miou = numpy_iou(gt_test_image,blur_map)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        perclass_accuracy_conf_matrix = confusion_matrix(gt_test_image.flatten(),
                                                         blur_map.flatten(),labels=[0,1,2,3,4],
                                                         normalize="true")
        # perclass_accuracy_conf_matrix_1 = confusion_matrix(gt_test_image.flatten(),
        #                                                  blur_map.flatten(), labels=[0, 1, 2, 3, 4])

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(5):
            if (perclass_accuracy_conf_matrix[lab,:] == 0).all() and (perclass_accuracy_conf_matrix[:,lab] == 0).all():
                pass
            else:
                classesList[lab].append(perclass_accuracy[lab])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(gt_test_image.flatten(),blur_map.flatten(), labels=[0,1,2,3,4],
                           average='micro')

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        f1score_list.append(f1score)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape)
        rgb_gt_map = np.zeros(test_image.shape)
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

# main test function
def test_with_muri_dataset_blur_noblur():
    print("MURI Testing")

    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample+'/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_blur_path, regx='/*.(png|PNG)', printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_gt_path, regx='/*.(png|PNG)', printable=False))

    ###Load Testing Data ####
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.muri_blur_path, n_threads=100, mode='RGB')
    test_mask_imgs = read_all_imgs(test_mask_img_list, path=config.TEST.muri_gt_path, n_threads=100, mode='RGB2GRAY2')

    test_classification_mask = []
    # print train_mask_imgs
    # img_n = 0
    for img in test_mask_imgs:
        tmp_class = img
        tmp_classification = np.concatenate((img, img, img), axis=2)

        # only checking if there is blur vs no blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
        tmp_class[np.where(tmp_classification[:, :, 0] == 64)] = 1  # motion blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 128)] = 1  # out of focus blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 192)] = 1  # darkness blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 255)] = 1  # brightness blur

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
            net_regression, m1, m2, m3, m1B, m2B, m3B, m4B = Decoder_Network_classification(input, n, f0, f0_1, f1_2,
                                                                                            f2_3, reuse=False,
                                                                                            scope=scope2)

    #output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs),axis=3),axis=3)
    output = tf.nn.softmax(net_regression.outputs)

    ### DEFINE LOSS ###
    # accuracy_run = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(output_map, classification_map))),
    #                 dtype=tf.float32)*(1/(h*w))
    #mean_iou, update_op = tf.compat.v1.metrics.mean_iou(labels=classification_map,predictions=predictions,num_classes=5)

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
    f1score_list = []
    classesList = [[],[]]
    # initalize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1','mIOU',
                         'f1_score','Total Num of Class 0','Total Num of Class 1','Total Num of Correct 0',
                         'Total Num of Class 1'])

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
        blurMap = sess.run([output],{net_regression.inputs: np.expand_dims(test_image, axis=0)})

        blurMap = np.squeeze(blurMap)
        gt_test_image = np.squeeze(gt_test_image)
        if tl.global_flag['uncertainty_label']:
            blur_map = np.zeros((h, w))
            blur_map[np.sum(blurMap[:,:] >= .4,axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:,:] >= .4,axis=2) == 1],
                                                                         axis=1)
            blur_map[blur_map >= 1] = 1
            # uncertainty labeling
            blur_map[np.sum(blurMap[:, :] >= .4, axis=2) != 1] = 5
        else:
            blur_map = np.argmax(blurMap,axis=2)
            blur_map[blur_map >= 1] = 1

        #np.save(save_dir_sample + '/raw_' + image_name.replace(".png", ".npy"), np.squeeze(blur_map))
        accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(),np.squeeze(blur_map).flatten(),normalize=True)

        # calculate mean intersection of union
        miou = numpy_iou(gt_test_image,blur_map)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        perclass_accuracy_conf_matrix = confusion_matrix(gt_test_image.flatten(),blur_map.flatten(),labels=[0,1],
                                                         normalize="true")
        perclass_accuracy_conf_matrix_1 = confusion_matrix(gt_test_image.flatten(),blur_map.flatten(), labels=[0, 1])

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(2):
            if (perclass_accuracy_conf_matrix[lab,:] == 0).all() and (perclass_accuracy_conf_matrix[:,lab] == 0).all():
                pass
            else:
                classesList[lab].append(perclass_accuracy[lab])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(gt_test_image.flatten(),blur_map.flatten(), labels=[0,1],
                           average='micro')

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        f1score_list.append(f1score)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape)
        rgb_gt_map = np.zeros(test_image.shape)
        # blue motion blur
        rgb_blur_map[blur_map == 1] = [255,0,0]
        rgb_gt_map[gt_test_image == 1] = [255,0,0]
        # green focus blur
        # rgb_blur_map[blur_map == 2] = [0, 255, 0]
        # rgb_gt_map[gt_test_image == 2] = [0, 255, 0]
        # # red darkness blur
        # rgb_blur_map[blur_map == 3] = [0, 0, 255]
        # rgb_gt_map[gt_test_image == 3] = [0, 0, 255]
        # # pink brightness blur
        # rgb_blur_map[blur_map == 4] = [255, 192, 203]
        # rgb_gt_map[gt_test_image == 4] = [255, 192, 203]
        # yellow unknown blur
        rgb_blur_map[blur_map == 5] = [0, 255, 255]

        log = "[*] Testing image name:"+image_name+" time: %4.4fs, Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
                                    "Accuracy for Class 1 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (time.time() - start_time,accuracy,perclass_accuracy[0],perclass_accuracy[1],miou,f1score)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image_name,str(np.round(accuracy,8)),str(np.round(perclass_accuracy[0],8)),
                   str(np.round(perclass_accuracy[1],8)),str(np.round(miou,8)),str(np.round(f1score,8)),
                       str(np.sum(gt_test_image == 0)),str(np.sum(gt_test_image == 1)),
                       str(perclass_accuracy_conf_matrix_1[0,0]),str(perclass_accuracy_conf_matrix_1[1,1])]
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
          "Max IoU: %.8f Variance: %.8f Max F1_score: %.8f\n" % (np.max(np.array(accuracy_list)),
           np.max(np.array(classesList[0]),axis=0),np.max(np.array(classesList[1]),axis=0),np.max(np.array(miou_list)),
                                                   np.var(np.asarray(accuracy_list)),np.max(np.array(f1score_list)))
    log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
          "Mean IoU: %.8f Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_list)), np.mean(np.array(classesList[0])),
           np.mean(np.array(classesList[1])),np.mean(np.array(miou_list)),np.mean(np.array(f1score_list)))
    # only way to write to log file while running
    with open(save_dir_sample + "/testing_metrics.log", "a") as f:
        # perform file operations
        f.write(log)
        f.write(log2)

def test_with_muri_dataset_origonal_model():
    print("MURI Testing")

    date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample+'/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_blur_path, regx='/*.(png|PNG)', printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.muri_gt_path, regx='/*.(png|PNG)', printable=False))

    ###Load Testing Data ####
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.muri_blur_path, n_threads=100, mode='RGB')
    test_mask_imgs = read_all_imgs(test_mask_img_list, path=config.TEST.muri_gt_path, n_threads=100, mode='RGB2GRAY2')

    test_classification_mask = []
    # print train_mask_imgs
    # img_n = 0
    for img in test_mask_imgs:
        tmp_class = img
        tmp_classification = np.concatenate((img, img, img), axis=2)

        tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
        tmp_class[np.where(tmp_classification[:, :, 0] == 64)] = 1  # motion blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 128)] = 2  # out of focus blur
        # don't have all labels need to modify labels to fit
        tmp_class[np.where(tmp_classification[:, :, 0] == 192)] = 0  # darkness blur make this focus blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 255)] = 0  # brightness blur

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
            net_regression, m1, m2, m3 = Decoder_Network_classification_3_labels(input, n, f0, f0_1, f1_2, f2_3,
                                                                                 reuse=False,scope=scope2)

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs),axis=3),axis=3)
    output = tf.nn.softmax(net_regression.outputs)

    ### DEFINE LOSS ###
    accuracy_run = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(output_map, classification_map))),
                    dtype=tf.float32)*(1/(h*w))
    #mean_iou, update_op = tf.compat.v1.metrics.mean_iou(labels=classification_map,predictions=predictions,num_classes=5)

    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Load checkpoint
    get_weights(sess, net_regression)
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file

    # file_name = './model/SA_net_{}.ckpt'.format(tl.global_flag['mode'])
    # reader = py_checkpoint_reader.NewCheckpointReader(file_name)
    # state_dict = {v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()}
    # get_weights_checkpoint(sess, net_regression, state_dict)

    accuracy_list = []
    miou_list = []
    f1score_list = []
    classesList = [[],[],[]]
    # initalize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1', 'Accuracy for Class 2',
                        'mIOU','f1_score'])

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
        blurMap, accuracy = sess.run([output, accuracy_run],
                                                {net_regression.inputs: np.expand_dims(test_image, axis=0),
                                                 classification_map: np.expand_dims(gt_test_image, axis=0)})
        blurMap = np.squeeze(blurMap)
        gt_test_image = np.squeeze(gt_test_image)
        if tl.global_flag['uncertainty_label']:
            blur_map = np.zeros((h, w))
            blur_map[np.sum(blurMap[:,:] >= .2,axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:,:] >= .2,axis=2) == 1],
                                                                         axis=1)
            # uncertainty labeling
            blur_map[np.sum(blurMap[:, :] >= .2, axis=2) != 1] = 5
        else:
            blur_map = np.argmax(blurMap,axis=2)
        # blur_map = np.zeros((h,w))
        # blur_map[np.sum(blurMap[:,:] >= .2,axis=2) == 1] = np.argmax(blurMap[np.sum(blurMap[:,:] >= .2,axis=2) == 1],
        #                                                              axis=1)
        # # uncertainty labeling
        # blur_map[np.sum(blurMap[:, :] >= .2, axis=2) != 1] = 5

        #np.save(save_dir_sample + '/raw_' + image_name.replace(".png", ".npy"), np.squeeze(blur_map))
        #accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(),np.squeeze(blur_map).flatten(),normalize=True)

        # calculate mean intersection of union
        miou = numpy_iou(gt_test_image,blur_map)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        perclass_accuracy_conf_matrix = confusion_matrix(gt_test_image.flatten(),
                                                         blur_map.flatten(),labels=[0,1,2],
                                                         normalize="true")

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        for lab in range(3):
            if (perclass_accuracy_conf_matrix[lab,:] == 0).all() and (perclass_accuracy_conf_matrix[:,lab] == 0).all():
                pass
            else:
                classesList[lab].append(perclass_accuracy[lab])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(gt_test_image.flatten(),blur_map.flatten(), labels=[0,1,2],
                           average='micro')

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        f1score_list.append(f1score)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape)
        rgb_gt_map = np.zeros(test_image.shape)
        # blue motion blur
        rgb_blur_map[blur_map == 1] = [255,0,0]
        rgb_gt_map[gt_test_image == 1] = [255,0,0]
        # green focus blur
        rgb_blur_map[blur_map == 2] = [0, 255, 0]
        rgb_gt_map[gt_test_image == 2] = [0, 255, 0]
        # red darkness blur
        # rgb_blur_map[blur_map == 3] = [0, 0, 255]
        # rgb_gt_map[gt_test_image == 3] = [0, 0, 255]
        # # pink brightness blur
        # rgb_blur_map[blur_map == 4] = [255, 192, 203]
        # rgb_gt_map[gt_test_image == 4] = [255, 192, 203]
        # # yellow unknown blur
        # rgb_blur_map[blur_map == 5] = [0, 255, 255]

        log = "[*] Testing image name:"+image_name+" time: %4.4fs, Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
                                    "Accuracy for Class 1 %.8f Accuracy for Class 2 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (time.time() - start_time,accuracy,perclass_accuracy[0],perclass_accuracy[1],perclass_accuracy[2],
                 miou,f1score)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image_name,str(np.round(accuracy,8)),str(np.round(perclass_accuracy[0],8)),
                   str(np.round(perclass_accuracy[1],8)),str(np.round(perclass_accuracy[2],8)),str(np.round(miou,8)),
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
          "Max Accuracy Class 2: %.8f Max IoU: %.8f " \
          "Variance: %.8f Max F1_score: %.8f\n" % (np.max(np.array(accuracy_list)),
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

def test_cuhk_dataset():
    print("Blurmap Testing")

    #date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample+'/gt')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary_result/')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary_result/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.cuhk_blur_path, regx='/*.(png|PNG)', printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.cuhk_gt_path, regx='/*.(png|PNG)', printable=False))
    # test_blur_img_list = sorted(
    #     tl.files.load_file_list(path=config.TEST.cuhk_blur_path, regx='/*.(jpg|JPG)', printable=False))
    # test_mask_img_list = sorted(
    #     tl.files.load_file_list(path=config.TEST.cuhk_gt_path, regx='/*.(jpg|JPG)', printable=False))

    ###Load Testing Data ####
    test_blur_imgs = read_all_imgs(test_blur_img_list, path=config.TEST.cuhk_blur_path, n_threads=100, mode='RGB')
    test_mask_imgs = read_all_imgs(test_mask_img_list, path=config.TEST.cuhk_gt_path, n_threads=100, mode='RGB2GRAY2')

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
    h = test_mask_imgs[0].shape[0]
    w = test_mask_imgs[0].shape[1]

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [1, h, w, 3], name='input_patches')
    labels = tf.compat.v1.placeholder('int64', [1, h, w, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3, m1B, m2B, m3B, m4B = Decoder_Network_classification(input, n, f0, f0_1, f1_2,
                                                                                            f2_3, reuse=False,
                                                                                            scope=scope2)

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs),axis=3),axis=3)
    output = tf.nn.softmax(net_regression.outputs)
    # output_map1 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m1.outputs),axis=3),axis=3)
    # output_map2 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m2.outputs),axis=3),axis=3)
    # output_map3 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(m3.outputs),axis=3),axis=3)

    ### DEFINE LOSS ###
    loss1 = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(output_map, labels))),
                    dtype=tf.float32)*(1/(h*w))
    #mean_iou, update_op = tf.compat.v1.metrics.mean_iou(labels=classification_map,predictions=predictions,num_classes=5)

    # Load checkpoint
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
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
    classesList = [[],[],[],[],[]]
    accuracy_list_binary = []
    miou_list_binary = []
    f1score_binary_list = []
    classesListBinary = [[], []]
    # initalize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1',
                         'Accuracy for Class 2','Accuracy for Class 3','Accuracy for Class 4','mIOU','f1_score',
                         'Number of Pixels Correct Class 0','Number of Pixels Correct Class 1',
                         'Number of Pixels Correct Class 2','Number of Pixels Correct Class 3',
                         'Number of Pixels Correct Class 4','Number of Pixels Incorrect Class 0',
                         'Number of Pixels Incorrect Class 1','Number of Pixels Incorrect Class 2',
                         'Number of Pixels Incorrect Class 3','Number of Pixels Incorrect Class 4'])

    with open(save_dir_sample + "/testing_metrics_binary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1','mIOU',
                         'f1_score','Number of Pixels Correct','Number of Pixels Incorrect'])
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
        # step run we run the network 100 times
        if tl.global_flag['uncertainty_label']:
            # option 1
            blurMap = np.squeeze(sess.run([output], {net_regression.inputs: np.expand_dims((test_image), axis=0)}))
            blur_map = np.zeros((h, w))
            # numpy array with labels
            blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) == 1] = np.argmax(
                blurMap[np.sum(blurMap[:, :] >= threshold, axis=2) == 1],
                axis=1)
            # uncertainty labeling
            blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) != 1] = 5
            if (blur_map == 5).shape[0] != 0:
                plt.rc('font', size=20)  # controls default text size
                plt.rc('axes', titlesize=20)  # fontsize of the title
                plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
                plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
                plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
                #plt.rc('legend', fontsize=30)  # fontsize of the legend
                plt.bar([0, 1, 2, 3, 4], blurMap[blur_map == 5][10], color='b', width=0.5)
                plt.xticks([0, 1, 2, 3, 4], ['No Blur', 'Motion', 'Focus','Darkness','Brightness'], rotation=45,
                           fontweight="bold")
                plt.yticks([0,0.1,0.2,0.3,0.4,0.5],fontweight="bold")
                plt.xlabel('Class Labels',fontweight="bold")
                plt.ylabel('Softmax Probability Output',fontweight="bold")
                # plt.title('Uncertain Label Example 1\n\n',
                #           fontweight="bold")
                plt.show()

            blur_map = np.expand_dims(blur_map[:,:,np.newaxis],axis=0)

            # option 2
            # blurMapList = []
            # for i in range(10):
            #     blurMapList.append(
            #         np.squeeze(sess.run([output], {net_regression.inputs: np.expand_dims((test_image), axis=0)})))
            # blur_map = np.zeros((h, w))
            # # blur_map = np.argmax(blurMap,axis=2)[:,:,np.newaxis].astype(np.int32)
            # # numpy array with labels
            # blurMap = np.exp(np.array(blurMapList))
            # percentBlurMap = np.percentile(blurMap, 50, axis=0)
            # maxthresholdIndex = np.max(percentBlurMap, axis=2)
            # blur_map[maxthresholdIndex >= threshold] = np.argmax(percentBlurMap >= threshold, axis=2)[
            #     maxthresholdIndex >= threshold]
            # blur_map[maxthresholdIndex < threshold] = 5
            # # blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) == 1] = np.argmax(
            # #     blurMap[np.sum(blurMap[:, :] >= threshold, axis=2) == 1],
            # #     axis=1)
            # # # uncertainty labeling
            # # blur_map[np.sum(blurMap[:, :] >= threshold, axis=2) != 1] = 5
            # blur_map = np.expand_dims(blur_map[:, :, np.newaxis], axis=0)

            # option 3
            # blurMap = np.squeeze(sess.run([output], {net_regression.inputs: np.expand_dims((test_image), axis=0)}))
            # blur_map = np.zeros((h, w))
            # # numpy array with labels
            # medianBlurMapSoftmax = np.median(blurMap, axis=2)
            # blur_map[medianBlurMapSoftmax <= threshold] = np.argmax(blurMap,axis=2)[medianBlurMapSoftmax <= threshold]
            # # uncertainty labeling
            # blur_map[medianBlurMapSoftmax > threshold] = 5
            # blur_map = np.expand_dims(blur_map[:,:,np.newaxis],axis=0)
        else:
            #blur_map,o1,o2,o3 = sess.run([output_map,output_map1,output_map2,output_map3],{net_regression.inputs: np.expand_dims((test_image), axis=0)})
            blur_map = sess.run([output_map],{net_regression.inputs: np.expand_dims((test_image), axis=0)})[0]

        accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(), np.squeeze(blur_map).flatten(), normalize=True)
        # compare binary map
        blur_map_binary = np.copy(blur_map)
        blur_map_binary[blur_map > 0] = 1

        accuracy_binary = sess.run([loss1], {output_map: blur_map_binary, labels: np.expand_dims((test_gt_image_blur_no_blur), axis=0)})[0]

        blur_map = np.squeeze(blur_map)
        blur_map_binary = np.squeeze(blur_map_binary)
        test_gt_image_blur_no_blur = np.squeeze(test_gt_image_blur_no_blur)
        gt_test_image = np.squeeze(gt_test_image)
        # o1 = np.squeeze(o1)
        # o2 = np.squeeze(o2)
        # o3 = np.squeeze(o3)
        #
        # calculate mean intersection of union
        miou = numpy_iou(gt_test_image, blur_map, 5)
        miou_binary = numpy_iou(test_gt_image_blur_no_blur, blur_map_binary, 2)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        # accuracy0 = accuracy_score(test_gt_image.flatten(),blur_map.flatten(),normalize="true")
        perclass_accuracy_conf_matrix = confusion_matrix(gt_test_image.flatten(), blur_map.flatten(), labels=[0, 1, 2,3,4],
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

        perclass_accuracy_conf_matrix_notnorm = confusion_matrix(gt_test_image.flatten(),blur_map.flatten(),
                                                         labels=[0, 1, 2,3,4])
        imagePixelsCorrectCount = np.diag(perclass_accuracy_conf_matrix_notnorm)
        imagePixelIncorrectCount = np.array([np.sum(perclass_accuracy_conf_matrix_notnorm[1:,0]),
                                                  np.sum(np.row_stack([np.array(
                                                      [perclass_accuracy_conf_matrix_notnorm[0,1]])[:,np.newaxis],
                                                        perclass_accuracy_conf_matrix_notnorm[2:,1][:,np.newaxis]])),
                                                  np.sum([perclass_accuracy_conf_matrix_notnorm[0:2, 2],
                                                          perclass_accuracy_conf_matrix_notnorm[3:, 2]]),
                                                  np.sum(np.row_stack([
                                                      perclass_accuracy_conf_matrix_notnorm[:3, 3][:,np.newaxis],
                                                      perclass_accuracy_conf_matrix_notnorm[4:, 3][:,np.newaxis]])),
                                             np.sum(perclass_accuracy_conf_matrix_notnorm[:4,4])])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(gt_test_image.flatten(), blur_map.flatten(), labels=[0, 1, 2,3,4], average='micro')
        f1score_binary = f1_score(test_gt_image_blur_no_blur.flatten(), blur_map_binary.flatten(), labels=[0, 1],
                                  average='micro')

        # record accuracy miou and f1 score in test set
        accuracy_list.append(accuracy)
        miou_list.append(miou)
        f1score_list.append(f1score)
        accuracy_list_binary.append(accuracy_binary)
        miou_list_binary.append(miou_binary)
        f1score_binary_list.append(f1score_binary)

        # only output every 100
        #if i % 100 == 0:

        blur_map = np.squeeze(blur_map)
        gt_map = np.squeeze(gt_test_image)

        # now color code
        rgb_blur_map = np.zeros(test_image.shape).astype(np.uint8)
        # rgb_blur_map_1 = np.zeros((o1.shape[0],o1.shape[1],3)).astype(np.uint8)
        # rgb_blur_map_2 = np.zeros((o2.shape[0], o2.shape[1], 3)).astype(np.uint8)
        # rgb_blur_map_3 = np.zeros((o3.shape[0], o3.shape[1], 3)).astype(np.uint8)
        rgb_gt_map = np.zeros(test_image.shape).astype(np.uint8)
        # rgb_gt_map_1 = np.zeros((o1.shape[0], o1.shape[1], 3)).astype(np.uint8)
        # rgb_gt_map_2 = np.zeros((o2.shape[0], o2.shape[1], 3)).astype(np.uint8)
        # rgb_gt_map_3 = np.zeros((o3.shape[0], o3.shape[1], 3)).astype(np.uint8)
        # gt_o1 = np.squeeze(cv2.resize(gt_test_image, (o1.shape[1], o1.shape[0]), interpolation=cv2.INTER_NEAREST))
        # gt_o2 = np.squeeze(cv2.resize(gt_test_image, (o2.shape[1], o2.shape[0]), interpolation=cv2.INTER_NEAREST))
        # gt_o3 = np.squeeze(cv2.resize(gt_test_image, (o3.shape[1], o3.shape[0]), interpolation=cv2.INTER_NEAREST))

        # red motion blur
        rgb_blur_map[blur_map == 1] = [255,0,0]
        # rgb_blur_map_1[o1 == 1] = [255, 0, 0]
        # rgb_blur_map_2[o2 == 1] = [255, 0, 0]
        # rgb_blur_map_3[o3 == 1] = [255, 0, 0]
        rgb_gt_map[gt_map == 1] = [255,0,0]
        # rgb_gt_map_1[gt_o1 == 1] = [255, 0, 0]
        # rgb_gt_map_2[gt_o2 == 1] = [255, 0, 0]
        # rgb_gt_map_3[gt_o3 == 1] = [255, 0, 0]
        # green focus blur
        rgb_blur_map[blur_map == 2] = [0, 255, 0]
        # rgb_blur_map_1[o1 == 2] = [0, 255, 0]
        # rgb_blur_map_2[o2 == 2] = [0, 255, 0]
        # rgb_blur_map_3[o3 == 2] = [0, 255, 0]
        rgb_gt_map[gt_map == 2] = [0, 255, 0]
        # rgb_gt_map_1[gt_o1 == 2] = [0, 255, 0]
        # rgb_gt_map_2[gt_o2 == 2] = [0, 255, 0]
        # rgb_gt_map_3[gt_o3 == 2] = [0, 255, 0]
        # blue darkness blur
        rgb_blur_map[blur_map == 3] = [0, 0, 255]
        # rgb_blur_map_1[o1 == 3] = [0, 0, 255]
        # rgb_blur_map_2[o2 == 3] = [0, 0, 255]
        # rgb_blur_map_3[o3 == 3] = [0, 0, 255]
        rgb_gt_map[gt_map == 3] = [0, 0, 255]
        # rgb_gt_map_1[gt_o1 == 3] = [0, 0, 255]
        # rgb_gt_map_2[gt_o2 == 3] = [0, 0, 255]
        # rgb_gt_map_3[gt_o3 == 3] = [0, 0, 255]
        # pink brightness blur
        rgb_blur_map[blur_map == 4] = [255, 192, 203]
        # rgb_blur_map_1[o1 == 4] = [255, 192, 203]
        # rgb_blur_map_2[o2 == 4] = [255, 192, 203]
        # rgb_blur_map_3[o3 == 4] = [255, 192, 203]
        rgb_gt_map[gt_map == 4] = [255, 192, 203]
        # rgb_gt_map_1[gt_o1 == 4] = [255, 192, 203]
        # rgb_gt_map_2[gt_o2 == 4] = [255, 192, 203]
        # rgb_gt_map_3[gt_o3 == 4] = [255, 192, 203]
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
                       str(np.round(perclass_accuracy[3],8)),str(np.round(perclass_accuracy[4],8)),
                       str(np.round(miou,8)),str(np.round(f1score,8)),str(imagePixelsCorrectCount[0]),
                       str(imagePixelsCorrectCount[1]),str(imagePixelsCorrectCount[2]),str(imagePixelsCorrectCount[3]),
                       str(imagePixelsCorrectCount[4]),str(imagePixelIncorrectCount[0]),
                       str(imagePixelIncorrectCount[1]),str(imagePixelIncorrectCount[2]),
                       str(imagePixelIncorrectCount[3]),str(imagePixelIncorrectCount[4])]
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
                       str(np.round(f1score, 8)),str(np.sum(imagePixelsCorrectCount)),
                       str(np.sum(imagePixelIncorrectCount))]
        with open(save_dir_sample + "/testing_metrics_binary.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(string_list)

        # only output every 100
        #if i % 100 == 0:
        if ".jpg" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            # imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            # imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            # imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".JPG" in image_name:
            image_name.replace(".JPG", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".JPG", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map)
            # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_1)
            # imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_1)
            # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_2)
            # imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_2)
            # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_3)
            # imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".JPG", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".JPG", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".PNG" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            # imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            # imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            # imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)
        if ".png" in image_name:
            image_name.replace(".jpg", ".png")
            imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
            imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
            # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
            # imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
            # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
            # imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
            # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
            # imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
            cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
            cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                        test_gt_image_blur_no_blur * 255)

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

    plt.rc('font', size=20)  # controls default text size
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=30)  # fontsize of the legend

    plt.clf()
    final_confmatrix = confusion_matrix(np.array(all_gt_img_results).flatten(), np.array(all_img_results).flatten(),
                                        labels=[0, 1, 2,3,4], normalize="true")
    np.save('all_labels_results_conf_matrix.npy', final_confmatrix)
    final_confmatrix = np.round(final_confmatrix,3)
    plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Motion', 'Focus','Darkness','Brightness']
    #plt.title('Our Weights - Test Data Confusion Matrix')
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
    cbar = plt.colorbar(ticks=[final_confmatrix.min(), final_confmatrix.max()])
    cbar.ax.set_yticklabels(['0.0', ' 1.0'])
    plt.savefig(save_dir_sample+'conf_matrix_all_labels.png')
    #plt.show()

    plt.clf()
    final_confmatrix = confusion_matrix(np.array(all_gt_binary_results).flatten(),
                                        np.array(all_img_binary_results).flatten(), labels=[0, 1], normalize="true")
    np.save('all_binary_results_conf_matrix.npy',final_confmatrix)
    final_confmatrix = np.round(final_confmatrix, 3)
    plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    classNames = ['No Blur', 'Blur']
    #plt.title('Our Weights - Test Data Confusion Matrix')
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
    cbar = plt.colorbar(ticks=[final_confmatrix.min(), final_confmatrix.max()])
    cbar.ax.set_yticklabels(['0.0', ' 1.0'])
    plt.savefig(save_dir_sample+'conf_matrix_binary.png')
    #plt.show()

    return 0

def test_cuhk_sensitivity_dataset():
    print("Blurmap Sensitivity Testing")

    #date = datetime.datetime.now().strftime("%y.%m.%d")
    save_dir_sample = 'output_sensitivity_{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_sample)
    tl.files.exists_or_mkdir(save_dir_sample+'/gt')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary_result/')
    tl.files.exists_or_mkdir(save_dir_sample + '/binary_result/gt')

    test_blur_img_list = sorted(tl.files.load_file_list(path=config.TEST.cuhk_blur_path, regx='/*.(png|PNG)', printable=False))
    test_mask_img_list = sorted(tl.files.load_file_list(path=config.TEST.cuhk_gt_path, regx='/*.(png|PNG)', printable=False))

    # need to make 3 weight classifications for all 3 sizes
    h1,w1 = 480, 640
    scale = 30
    width = int(480 * scale / 100)
    height = int(640 * scale / 100)
    h2,w2 = (width, height)
    scale = 60
    width = int(480 * scale / 100)
    height = int(640 * scale / 100)
    h3,w3 = (width, height)
    ### DEFINE MODEL ###
    patches_blurred1 = tf.compat.v1.placeholder('float32', [1, h1, w1, 3], name='input_patches')
    labels1 = tf.compat.v1.placeholder('int64', [1, h1, w1, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input1, n1, f01, f0_11, f1_21, f2_31 = VGG19_pretrained(patches_blurred1, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression1, _, _, _, _, _, _, _ = Decoder_Network_classification(input1, n1, f01, f0_11, f1_21,
                                                                                            f2_31, reuse=False,
                                                                                            scope=scope2)
    output_map1 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression1.outputs),axis=3),axis=3)
    #output1 = tf.nn.softmax(net_regression1.outputs)

    ### DEFINE ACCURACY ###
    loss1 = tf.cast(tf.math.reduce_sum(1-tf.math.abs(tf.math.subtract(output_map1, labels1))),
                    dtype=tf.float32)*(1/(h1*w1))

    patches_blurred2 = tf.compat.v1.placeholder('float32', [1, h2, w2, 3], name='input_patches')
    labels2 = tf.compat.v1.placeholder('int64', [1, h2, w2, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input2, n2, f02, f0_12, f1_22, f2_32 = VGG19_pretrained(patches_blurred2, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression2, _, _, _, _, _, _, _ = Decoder_Network_classification(input2, n2, f02, f0_12, f1_22,
                                                                                            f2_32, reuse=False,
                                                                                            scope=scope2)
    output_map2 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression2.outputs), axis=3), axis=3)
    #output2 = tf.nn.softmax(net_regression2.outputs)

    ### DEFINE ACCURACY ###
    loss2 = tf.cast(tf.math.reduce_sum(1 - tf.math.abs(tf.math.subtract(output_map2, labels2))),
                    dtype=tf.float32) * (1 / (h2 * w2))

    patches_blurred3 = tf.compat.v1.placeholder('float32', [1, h3, w3, 3], name='input_patches')
    labels3 = tf.compat.v1.placeholder('int64', [1, h3, w3, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input3, n3, f03, f0_13, f1_23, f2_33 = VGG19_pretrained(patches_blurred3, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression3, _, _, _, _, _, _, _ = Decoder_Network_classification(input3, n3, f03, f0_13, f1_23,
                                                                                            f2_33, reuse=False,
                                                                                            scope=scope2)
    output_map3 = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression3.outputs), axis=3), axis=3)
    #output3 = tf.nn.softmax(net_regression3.outputs)

    ### DEFINE ACCURACY ###
    loss3 = tf.cast(tf.math.reduce_sum(1 - tf.math.abs(tf.math.subtract(output_map3, labels3))),
                    dtype=tf.float32) * (1 / (h3 * w3))

    # Load checkpoint
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
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
    get_weights_checkpoint(sess, net_regression1, state_dict)
    get_weights_checkpoint(sess, net_regression2, state_dict)
    get_weights_checkpoint(sess, net_regression3, state_dict)
    state_dict,reader = None,None

    net_regression1.test()
    net_regression2.test()
    net_regression3.test()

    # accuracy_list = []
    # miou_list = []
    # f1score_list = []
    # classesList = [[],[],[],[],[]]
    # accuracy_list_binary = []
    # miou_list_binary = []
    # f1score_binary_list = []
    # classesListBinary = [[], []]
    # initalize the csv metrics output
    with open(save_dir_sample + "/testing_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1',
                         'Accuracy for Class 2','Accuracy for Class 3','Accuracy for Class 4','mIOU','f1_score',
                         'Number of Pixels Correct Class 0','Number of Pixels Correct Class 1',
                         'Number of Pixels Correct Class 2','Number of Pixels Correct Class 3',
                         'Number of Pixels Correct Class 4','Number of Pixels Incorrect Class 0',
                         'Number of Pixels Incorrect Class 1','Number of Pixels Incorrect Class 2',
                         'Number of Pixels Incorrect Class 3','Number of Pixels Incorrect Class 4'])

    with open(save_dir_sample + "/testing_metrics_binary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Overall Accuracy', 'Accuracy for Class 0', 'Accuracy for Class 1','mIOU',
                         'f1_score','Number of Pixels Correct','Number of Pixels Incorrect'])
    f = None
    writer = None

    # all_img_results = []
    # all_gt_img_results = []
    # all_img_binary_results = []
    # all_gt_binary_results = []

    for i in range(len(test_blur_img_list)):
        ###Load Testing Data ####
        test_image = get_imgs_RGB_fn(test_blur_img_list[i],config.TEST.cuhk_blur_path)
        gt_test_image = get_imgs_RGBGRAY_2_fn(test_mask_img_list[i],config.TEST.cuhk_gt_path)
        tmp_class = np.copy(gt_test_image)
        tmp_class[gt_test_image == 0] = 0  # sharp
        tmp_class[gt_test_image == 64] = 1  # motion blur
        tmp_class[gt_test_image == 128] = 2  # out of focus blur
        tmp_class[gt_test_image == 192] = 3  # darkness blur
        tmp_class[gt_test_image == 255] = 4  # brightness blur
        gt_test_image = tmp_class
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
        height = test_image.shape[0]

        # Model
        if height == h1:
            blur_map = sess.run([output_map1], {net_regression1.inputs: np.expand_dims((test_image), axis=0)})[0]

            accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(), np.squeeze(blur_map).flatten(),
                                      normalize=True)
            # compare binary map
            blur_map_binary = np.copy(blur_map)
            blur_map_binary[blur_map > 0] = 1

            accuracy_binary = sess.run([loss1], {output_map1: blur_map_binary,
                                                 labels1: np.expand_dims((test_gt_image_blur_no_blur), axis=0)})[0]
        elif height == h2:
            blur_map = sess.run([output_map2], {net_regression2.inputs: np.expand_dims((test_image), axis=0)})[0]

            accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(), np.squeeze(blur_map).flatten(),
                                      normalize=True)
            # compare binary map
            blur_map_binary = np.copy(blur_map)
            blur_map_binary[blur_map > 0] = 1

            accuracy_binary = sess.run([loss2], {output_map2: blur_map_binary,
                                                 labels2: np.expand_dims((test_gt_image_blur_no_blur), axis=0)})[0]
        else:
            blur_map = sess.run([output_map3], {net_regression3.inputs: np.expand_dims((test_image), axis=0)})[0]

            accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(), np.squeeze(blur_map).flatten(),
                                      normalize=True)
            # compare binary map
            blur_map_binary = np.copy(blur_map)
            blur_map_binary[blur_map > 0] = 1

            accuracy_binary = sess.run([loss3], {output_map3: blur_map_binary,
                                                 labels3: np.expand_dims((test_gt_image_blur_no_blur), axis=0)})[0]
        # uncertain labeling
        # step run we run the network 100 times
        # if tl.global_flag['uncertainty_label']:
        #     blurMap = []
        #     # need to figure out how to run this faster maybe (this is only for testing though)
        #     for i in range(100):
        #         blurMap.append(np.squeeze(sess.run([output], {net_regression.inputs: np.expand_dims((test_image), axis=0)})))
        #
        #     blur_map = np.zeros((256, 256))
        #     # find the percetile for each pixel
        #     prob = np.percentile(blurMap, 50, axis=0)
        #     # anything above the threshold is just thr argmax of the probabilities
        #     blur_map[prob > .2] = np.argmax(blurMap[prob > .2], axis=1)
        #     # uncertanity labeling
        #     blur_map[prob <= .2] = 5
        #     accuracy = accuracy_score(np.squeeze(gt_test_image).flatten(), np.squeeze(blur_map).flatten(),
        #                               normalize=True)
        # else:
            #blur_map,o1,o2,o3 = sess.run([output_map,output_map1,output_map2,output_map3],{net_regression.inputs: np.expand_dims((test_image), axis=0)})

        blur_map = np.squeeze(blur_map)
        blur_map_binary = np.squeeze(blur_map_binary)
        test_gt_image_blur_no_blur = np.squeeze(test_gt_image_blur_no_blur)
        gt_test_image = np.squeeze(gt_test_image)
        # o1 = np.squeeze(o1)
        # o2 = np.squeeze(o2)
        # o3 = np.squeeze(o3)
        #
        # calculate mean intersection of union
        miou = numpy_iou(gt_test_image, blur_map, 5)
        miou_binary = numpy_iou(test_gt_image_blur_no_blur, blur_map_binary, 2)

        # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
        # accuracy0 = accuracy_score(test_gt_image.flatten(),blur_map.flatten(),normalize="true")
        perclass_accuracy_conf_matrix = confusion_matrix(gt_test_image.flatten(), blur_map.flatten(), labels=[0, 1, 2,3,4],
                                                         normalize="true")
        # all_img_results.append(blur_map.flatten())
        # all_gt_img_results.append(gt_test_image.flatten())
        # all_img_binary_results.append(blur_map_binary.flatten())
        # all_gt_binary_results.append(test_gt_image_blur_no_blur.flatten())

        perclass_accuracy = perclass_accuracy_conf_matrix.diagonal()
        # for lab in range(5):
        #     if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (
        #             perclass_accuracy_conf_matrix[:, lab] == 0).all():
        #         pass
        #     else:
        #         classesList[lab].append(perclass_accuracy[lab])

        perclass_accuracy_conf_matrix = confusion_matrix(test_gt_image_blur_no_blur.flatten(),
                                                         blur_map_binary.flatten(),
                                                         labels=[0, 1], normalize="true")

        perclass_accuracy_binary = perclass_accuracy_conf_matrix.diagonal()
        # for lab in range(2):
        #     if (perclass_accuracy_conf_matrix[lab, :] == 0).all() and (
        #             perclass_accuracy_conf_matrix[:, lab] == 0).all():
        #         pass
        #     else:
        #         classesListBinary[lab].append(perclass_accuracy_binary[lab])

        perclass_accuracy_conf_matrix_notnorm = confusion_matrix(gt_test_image.flatten(),blur_map.flatten(),
                                                         labels=[0, 1, 2,3,4])
        imagePixelsCorrectCount = np.diag(perclass_accuracy_conf_matrix_notnorm)
        imagePixelIncorrectCount = np.array([np.sum(perclass_accuracy_conf_matrix_notnorm[1:,0]),
                                                  np.sum(np.row_stack([np.array(
                                                      [perclass_accuracy_conf_matrix_notnorm[0,1]])[:,np.newaxis],
                                                        perclass_accuracy_conf_matrix_notnorm[2:,1][:,np.newaxis]])),
                                                  np.sum([perclass_accuracy_conf_matrix_notnorm[0:2, 2],
                                                          perclass_accuracy_conf_matrix_notnorm[3:, 2]]),
                                                  np.sum(np.row_stack([
                                                      perclass_accuracy_conf_matrix_notnorm[:3, 3][:,np.newaxis],
                                                      perclass_accuracy_conf_matrix_notnorm[4:, 3][:,np.newaxis]])),
                                             np.sum(perclass_accuracy_conf_matrix_notnorm[:4,4])])

        # calculate f1 score
        # https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
        f1score = f1_score(gt_test_image.flatten(), blur_map.flatten(), labels=[0, 1, 2,3,4], average='micro')
        f1score_binary = f1_score(test_gt_image_blur_no_blur.flatten(), blur_map_binary.flatten(), labels=[0, 1],
                                  average='micro')

        # # record accuracy miou and f1 score in test set
        # accuracy_list.append(accuracy)
        # miou_list.append(miou)
        # f1score_list.append(f1score)
        # accuracy_list_binary.append(accuracy_binary)
        # miou_list_binary.append(miou_binary)
        # f1score_binary_list.append(f1score_binary)

        # for the sensotivity analysis so many images are going through we dont want to output all image outputs rather
        # output every 500
        if i % 500 == 0:
            blur_map = np.squeeze(blur_map)
            gt_map = np.squeeze(gt_test_image)
            # now color code
            rgb_blur_map = np.zeros(test_image.shape).astype(np.uint8)
            # rgb_blur_map_1 = np.zeros((o1.shape[0],o1.shape[1],3)).astype(np.uint8)
            # rgb_blur_map_2 = np.zeros((o2.shape[0], o2.shape[1], 3)).astype(np.uint8)
            # rgb_blur_map_3 = np.zeros((o3.shape[0], o3.shape[1], 3)).astype(np.uint8)
            rgb_gt_map = np.zeros(test_image.shape).astype(np.uint8)
            # rgb_gt_map_1 = np.zeros((o1.shape[0], o1.shape[1], 3)).astype(np.uint8)
            # rgb_gt_map_2 = np.zeros((o2.shape[0], o2.shape[1], 3)).astype(np.uint8)
            # rgb_gt_map_3 = np.zeros((o3.shape[0], o3.shape[1], 3)).astype(np.uint8)
            # gt_o1 = np.squeeze(cv2.resize(gt_test_image, (o1.shape[1], o1.shape[0]), interpolation=cv2.INTER_NEAREST))
            # gt_o2 = np.squeeze(cv2.resize(gt_test_image, (o2.shape[1], o2.shape[0]), interpolation=cv2.INTER_NEAREST))
            # gt_o3 = np.squeeze(cv2.resize(gt_test_image, (o3.shape[1], o3.shape[0]), interpolation=cv2.INTER_NEAREST))
            # red motion blur
            rgb_blur_map[blur_map == 1] = [255,0,0]
            # rgb_blur_map_1[o1 == 1] = [255, 0, 0]
            # rgb_blur_map_2[o2 == 1] = [255, 0, 0]
            # rgb_blur_map_3[o3 == 1] = [255, 0, 0]
            rgb_gt_map[gt_map == 1] = [255,0,0]
            # rgb_gt_map_1[gt_o1 == 1] = [255, 0, 0]
            # rgb_gt_map_2[gt_o2 == 1] = [255, 0, 0]
            # rgb_gt_map_3[gt_o3 == 1] = [255, 0, 0]
            # green focus blur
            rgb_blur_map[blur_map == 2] = [0, 255, 0]
            # rgb_blur_map_1[o1 == 2] = [0, 255, 0]
            # rgb_blur_map_2[o2 == 2] = [0, 255, 0]
            # rgb_blur_map_3[o3 == 2] = [0, 255, 0]
            rgb_gt_map[gt_map == 2] = [0, 255, 0]
            # rgb_gt_map_1[gt_o1 == 2] = [0, 255, 0]
            # rgb_gt_map_2[gt_o2 == 2] = [0, 255, 0]
            # rgb_gt_map_3[gt_o3 == 2] = [0, 255, 0]
            # blue darkness blur
            rgb_blur_map[blur_map == 3] = [0, 0, 255]
            # rgb_blur_map_1[o1 == 3] = [0, 0, 255]
            # rgb_blur_map_2[o2 == 3] = [0, 0, 255]
            # rgb_blur_map_3[o3 == 3] = [0, 0, 255]
            rgb_gt_map[gt_map == 3] = [0, 0, 255]
            # rgb_gt_map_1[gt_o1 == 3] = [0, 0, 255]
            # rgb_gt_map_2[gt_o2 == 3] = [0, 0, 255]
            # rgb_gt_map_3[gt_o3 == 3] = [0, 0, 255]
            # pink brightness blur
            rgb_blur_map[blur_map == 4] = [255, 192, 203]
            # rgb_blur_map_1[o1 == 4] = [255, 192, 203]
            # rgb_blur_map_2[o2 == 4] = [255, 192, 203]
            # rgb_blur_map_3[o3 == 4] = [255, 192, 203]
            rgb_gt_map[gt_map == 4] = [255, 192, 203]
            # rgb_gt_map_1[gt_o1 == 4] = [255, 192, 203]
            # rgb_gt_map_2[gt_o2 == 4] = [255, 192, 203]
            # rgb_gt_map_3[gt_o3 == 4] = [255, 192, 203]
            # yellow unknown blur
            rgb_blur_map[blur_map == 5] = [0, 255, 255]

        log = "[*] Testing image name:"+image_name+" Overall Accuracy: %.8f Accuracy for Class 0 %.8f " \
                                    "Accuracy for Class 1 %.8f Accuracy for Class 2 %.8f Accuracy for Class 3 %.8f " \
                                    "Accuracy for Class 4 %.8f mIOU: %.8f f1_score: %.8f\n" \
              % (accuracy,perclass_accuracy[0],perclass_accuracy[1],perclass_accuracy[2],perclass_accuracy[3],
                 perclass_accuracy[4],miou,f1score)
        # only way to write to log file while running
        with open(save_dir_sample + "/testing_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        # write csv file output for plots making
        string_list = [image_name,str(np.round(accuracy,8)),str(np.round(perclass_accuracy[0],8)),
                       str(np.round(perclass_accuracy[1],8)),str(np.round(perclass_accuracy[2],8)),
                       str(np.round(perclass_accuracy[3],8)),str(np.round(perclass_accuracy[4],8)),
                       str(np.round(miou,8)),str(np.round(f1score,8)),str(imagePixelsCorrectCount[0]),
                       str(imagePixelsCorrectCount[1]),str(imagePixelsCorrectCount[2]),str(imagePixelsCorrectCount[3]),
                       str(imagePixelsCorrectCount[4]),str(imagePixelIncorrectCount[0]),
                       str(imagePixelIncorrectCount[1]),str(imagePixelIncorrectCount[2]),
                       str(imagePixelIncorrectCount[3]),str(imagePixelIncorrectCount[4])]
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
                       str(np.round(f1score, 8)),str(np.sum(imagePixelsCorrectCount)),
                       str(np.sum(imagePixelIncorrectCount))]
        with open(save_dir_sample + "/testing_metrics_binary.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(string_list)
        if i % 500 == 0:
            if ".jpg" in image_name:
                image_name.replace(".jpg", ".png")
                imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
                imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
                # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
                # imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
                # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
                # imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
                # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
                # imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
                cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
                cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                            test_gt_image_blur_no_blur * 255)
            if ".JPG" in image_name:
                image_name.replace(".JPG", ".png")
                imageio.imwrite(save_dir_sample + '/' + image_name.replace(".JPG", ".png"), rgb_blur_map)
                imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map)
                # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_1)
                # imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_1)
                # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_2)
                # imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_2)
                # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".JPG", ".png"), rgb_blur_map_3)
                # imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".JPG", ".png"), rgb_gt_map_3)
                cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".JPG", ".png"), blur_map_binary * 255)
                cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".JPG", ".png"),
                            test_gt_image_blur_no_blur * 255)
            if ".PNG" in image_name:
                image_name.replace(".jpg", ".png")
                imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
                imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
                # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
                # imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
                # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
                # imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
                # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
                # imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
                cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
                cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                            test_gt_image_blur_no_blur * 255)
            if ".png" in image_name:
                image_name.replace(".jpg", ".png")
                imageio.imwrite(save_dir_sample + '/' + image_name.replace(".jpg", ".png"), rgb_blur_map)
                imageio.imwrite(save_dir_sample + '/gt/gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map)
                # imageio.imwrite(save_dir_sample + '/m1_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_1)
                # imageio.imwrite(save_dir_sample + '/gt/m1_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_1)
                # imageio.imwrite(save_dir_sample + '/m2_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_2)
                # imageio.imwrite(save_dir_sample + '/gt/m2_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_2)
                # imageio.imwrite(save_dir_sample + '/m3_results' + image_name.replace(".jpg", ".png"), rgb_blur_map_3)
                # imageio.imwrite(save_dir_sample + '/gt/m3_results_gt_' + image_name.replace(".jpg", ".png"), rgb_gt_map_3)
                cv2.imwrite(save_dir_sample + '/binary_result/' + image_name.replace(".jpg", ".png"), blur_map_binary * 255)
                cv2.imwrite(save_dir_sample + '/binary_result/gt/' + image_name.replace(".jpg", ".png"),
                            test_gt_image_blur_no_blur * 255)

    sess.close()
    # log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f " \
    #       "Max Accuracy Class 2: %.8f Max Accuracy Class 3: %.8f Max Accuracy Class 4: %.8f Max IoU: %.8f " \
    #       "Variance: %.8f Max F1_score: %.8f\n" % (np.max(np.array(accuracy_list)),
    #        np.max(np.array(classesList[0]),axis=0),np.max(np.array(classesList[1]),axis=0),
    #        np.max(np.array(classesList[2]),axis=0),np.max(np.array(classesList[3]),axis=0),
    #        np.max(np.array(classesList[4]),axis=0),np.max(np.array(miou_list)),np.var(np.asarray(accuracy_list)),
    #        np.max(np.array(f1score_list)))
    # log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
    #       "Mean Accuracy Class 2: %.8f Mean Accuracy Class 3: %.8f Mean Accuracy Class 4: %.8f Mean IoU: %.8f " \
    #       "Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_list)), np.mean(np.array(classesList[0])),
    #        np.mean(np.array(classesList[1])),np.mean(np.array(classesList[2])),np.mean(np.array(classesList[3])),
    #        np.mean(np.array(classesList[4])),np.mean(np.array(miou_list)),np.mean(np.array(f1score_list)))
    # # only way to write to log file while running
    # with open(save_dir_sample + "/testing_metrics.log", "a") as f:
    #     # perform file operations
    #     f.write(log)
    #     f.write(log2)
    #
    # log = "[*] Testing Max Overall Accuracy: %.8f Max Accuracy Class 0: %.8f Max Accuracy Class 1: %.8f Max IoU: %.8f " \
    #       "Variance: %.8f Max F1_score: %.8f\n" % (
    #           np.max(np.array(accuracy_list_binary)),
    #           np.max(np.array(classesListBinary[0]), axis=0), np.max(np.array(classesListBinary[1]), axis=0),
    #           np.max(np.array(miou_list_binary)), np.var(np.asarray(accuracy_list_binary)),
    #           np.max(np.array(f1score_binary_list)))
    # log2 = "[*] Testing Mean Overall Accuracy: %.8f Mean Accuracy Class 0: %.8f Mean Accuracy Class 1: %.8f " \
    #        "Mean IoU: %.8f Mean F1_score: %.8f\n" % (np.mean(np.array(accuracy_list_binary)),
    #                                                  np.mean(np.array(classesListBinary[0])),
    #                                                  np.mean(np.array(classesListBinary[1])),
    #                                                  np.mean(np.array(miou_list_binary)),
    #                                                  np.mean(np.array(f1score_binary_list)))
    # # only way to write to log file while running
    # with open(save_dir_sample + "/testing_metrics_binary.log", "a") as f:
    #     # perform file operations
    #     f.write(log)
    #     f.write(log2)

    # throw everything that is not used in trash
    # f1score_binary_list,accuracy_list = None,None
    # accuracy_list_binary,writer = None,None
    # classesListBinary = None
    # miou_list_binary = None
    # sess = None
    # classesList,perclass_accuracy = None,None
    # miou_list,perclass_accuracy_conf_matrix_notnorm = None,None
    # f1score_list,rgb_blur_map,blur_map,gt_map,rgb_gt_map = None,None,None,None,None

    # no way this will not fill memory need to just use excel for this
    # plt.clf()
    # final_confmatrix = confusion_matrix(np.array(all_gt_img_results).flatten(), np.array(all_img_results).flatten(),
    #                                     labels=[0, 1, 2,3,4], normalize="true")
    # np.save('all_labels_results_conf_matrix.npy', final_confmatrix)
    # final_confmatrix = np.round(final_confmatrix,3)
    # plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    # classNames = ['No Blur', 'Motion', 'Focus','Darkness','Brightness']
    # plt.title('Our Weights - Test Data Confusion Matrix')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # tick_marks = np.arange(len(classNames))
    # plt.xticks(tick_marks, classNames, rotation=45)
    # plt.yticks(tick_marks, classNames)
    # # for i in range(len(classNames)):
    # #     for j in range(len(classNames)):
    # #         plt.text(j, i, str(final_confmatrix[i][j]))
    # thresh = final_confmatrix.max() / 2.
    # for i in range(final_confmatrix.shape[0]):
    #     for j in range(final_confmatrix.shape[1]):
    #         plt.text(j, i, format(final_confmatrix[i, j]),
    #                  ha="center", va="center",
    #                  color="white" if final_confmatrix[i, j] > thresh else "black")
    # plt.tight_layout()
    # plt.colorbar()
    # plt.savefig(save_dir_sample+'conf_matrix_all_labels.png')
    # #plt.show()
    #
    # plt.clf()
    # final_confmatrix = confusion_matrix(np.array(all_gt_binary_results).flatten(),
    #                                     np.array(all_img_binary_results).flatten(), labels=[0, 1], normalize="true")
    # np.save('all_binary_results_conf_matrix.npy',final_confmatrix)
    # final_confmatrix = np.round(final_confmatrix, 3)
    # plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    # classNames = ['No Blur', 'Blur']
    # plt.title('Our Weights - Test Data Confusion Matrix')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # tick_marks = np.arange(len(classNames))
    # plt.xticks(tick_marks, classNames, rotation=45)
    # plt.yticks(tick_marks, classNames)
    # # for i in range(len(classNames)):
    # #     for j in range(len(classNames)):
    # #         plt.text(j, i, str(final_confmatrix[i][j]))
    # thresh = final_confmatrix.max() / 2.
    # for i in range(final_confmatrix.shape[0]):
    #     for j in range(final_confmatrix.shape[1]):
    #         plt.text(j, i, format(final_confmatrix[i, j]),
    #                  ha="center", va="center",
    #                  color="white" if final_confmatrix[i, j] > thresh else "black")
    # plt.tight_layout()
    # plt.colorbar()
    # plt.savefig(save_dir_sample+'conf_matrix_binary.png')
    # #plt.show()
    return 0
