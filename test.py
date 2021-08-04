# coding=utf-8
import csv
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorlayer as tl
import numpy as np
import math
from tensorflow.python.training import py_checkpoint_reader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from config import config
from utils import read_all_imgs
from model import Decoder_Network_classification, VGG19_pretrained
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
                                                         blur_map.flatten(),labels=[0,1,2,3,4],
                                                         normalize="true")

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

