# coding=utf-8
import csv

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from config import log_config
from utils import *
from model import *
import time
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

h = config.TRAIN.height
w = config.TRAIN.width

# https://izziswift.com/better-way-to-shuffle-two-numpy-arrays-in-unison/
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

# updated chuk training with python 3.8 tensorflow 2.0
def train_with_CHUK():
    checkpoint_dir = "test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    input_path = config.TRAIN.blur_path
    gt_path = config.TRAIN.gt_path
    list_file = 'dataset/train_list.txt'
    if '.jpg' in tl.global_flag['image_extension']:
        train_blur_img_list = sorted(
        tl.files.load_file_list(path=input_path, regx='(out_of_focus|motion).*.(jpg|JPG)', printable=False))
    else:
        train_blur_img_list = sorted(
            tl.files.load_file_list(path=input_path, regx='(out_of_focus|motion).*.(png|PNG)', printable=False))
    train_mask_img_list = []

    for name in train_blur_img_list:
        if ".jpg" in name:
            train_mask_img_list.append(name.replace(".jpg", ".png"))
        else:
            train_mask_img_list.append(name.replace(".JPG", ".png"))

    ###Load Training Data ####
    blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=batch_size, mode='RGB')
    mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=batch_size, mode='GRAY')

    train_blur_imgs = []
    train_mask_imgs = []
    
    # get the chuk image list that is used for training 
    with open(list_file) as f:
        lines = f.readlines()
    indexs = np.arange(0,len(blur_imgs))
    testList = '\t'.join(lines)
    for idx in indexs:
        imagename = train_blur_img_list[idx]
        if imagename in testList:
            train_blur_imgs.append(blur_imgs[idx])
            train_mask_imgs.append(mask_imgs[idx])

    index = 0
    train_classification_mask = []
    # img_n = 0
    for img in train_mask_imgs:
        if index < 236: 
            tmp_class = img
            tmp_classification = np.concatenate((img, img, img), axis=2)
            tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
            tmp_class[np.where(tmp_classification[:, :, 0] > 0)] = 1  # defocus blur
        else:
            tmp_class = img
            tmp_classification = np.concatenate((img, img, img), axis=2)
            tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
            tmp_class[np.where(tmp_classification[:, :, 0] > 0)] = 2  # defocus blur

        train_classification_mask.append(tmp_class)
        index = index + 1
    
    train_classification_mask = np.array(train_classification_mask)
    train_blur_imgs = np.array(train_blur_imgs)
    print("Number of training images " + str(len(train_blur_imgs)))

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [batch_size, h, w, 3], name='input_patches')
    classification_map = tf.compat.v1.placeholder('int32', [batch_size, h, w, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3 = Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3, 
            reuse=False,scope=scope2)

    ### DEFINE LOSS ###
    loss1 = tl.cost.cross_entropy(net_regression.outputs, tf.squeeze(classification_map), name='loss1')
    loss2 = tl.cost.cross_entropy(m1.outputs,
                                  tf.squeeze(tf.image.resize(classification_map, [int(h / 2), int(w / 2)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss2')
    loss3 = tl.cost.cross_entropy(m2.outputs,
                                  tf.squeeze(tf.image.resize(classification_map, [int(h / 4), int(w / 4)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss3')
    loss4 = tl.cost.cross_entropy(m3.outputs,
                                  tf.squeeze(tf.image.resize(classification_map, [int(h / 8), int(w / 8)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss4')
    out = net_regression.outputs
    loss = loss1 + loss2 + loss3 + loss4

    with tf.compat.v1.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init * 0.1 * 0.1, trainable=False)
        lr_v2 = tf.Variable(lr_init * 0.1, trainable=False)

    ### DEFINE OPTIMIZER ###
    a_vars = tl.layers.get_variables_with_name('Unified', False, True)  # Unified
    var_list1 = tl.layers.get_variables_with_name('VGG', True, True)  # ?
    var_list2 = tl.layers.get_variables_with_name('UNet', True, True)  # ?
    opt1 = tf.optimizers.Adam(learning_rate=lr_v)
    opt2 = tf.optimizers.Adam(learning_rate=lr_v2)
    grads = tf.gradients(ys=loss, xs=var_list1 + var_list2, unconnected_gradients='zero')
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op = tf.group(train_op1, train_op2)

    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    # uncomment if on a gpu machine
    if tl.global_flag['gpu']:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        
            except RuntimeError as e:
                print(e)
        #####################################
    sess = tf.compat.v1.Session(config=configTf)

    print("initializing global variable...")
    sess.run(tf.compat.v1.global_variables_initializer())
    print("initializing global variable...DONE")

    ### LOAD VGG else load previous weights from starting from###
    if tl.global_flag['start_from'] != 0:
        print("loading initial checkpoint")
        tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                           save_dir=checkpoint_dir, var_list=a_vars, is_latest=True, printable=True)
    else:
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1',allow_pickle=True).item()

        params = []
        count_layers = 0
        for val in sorted(npz.items()):
            if count_layers < 16:
                W = np.asarray(val[1][0])
                b = np.asarray(val[1][1])
                print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
                params.extend([W, b])
            count_layers += 1

        sess.run(tl.files.assign_weights(params, n))

    ### START TRAINING ###
    augmentation_list = [0, 1]

    # initialize the csv metrics output
    with open(checkpoint_dir + "/training_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'total_error'])

    for epoch in range(tl.global_flag['start_from'], n_epoch + 1):
        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            sess.run(tf.compat.v1.assign(lr_v, lr_v * lr_decay))
            sess.run(tf.compat.v1.assign(lr_v2, lr_v2 * lr_decay))
            log = " ** new learning rate for Encoder %f\n" % (sess.run(lr_v))
            with open(checkpoint_dir + "/training_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
            log = " ** new learning rate for Decoder %f\n" % (sess.run(lr_v2))
            with open(checkpoint_dir + "/training_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
        elif epoch == tl.global_flag['start_from']:
            log = " ** init lr for Decoder: %f decay_every_init: %d, lr_decay: %f\n" % (sess.run(lr_v2), decay_every,
                                                                                        lr_decay)
            with open(checkpoint_dir + "/training_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
            log = " ** init lr for Encoder: %f decay_every_init: %d, lr_decay: %f\n" % (sess.run(lr_v), decay_every,
                                                                                        lr_decay)
            # print(log)
            with open(checkpoint_dir + "/training_metrics.log", "a") as f:
                # perform file operations
                f.write(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0

        # data shuffle***
        train_blur_imgs, train_classification_mask = unison_shuffled_copies(train_blur_imgs, train_classification_mask)

        for idx in range(0, len(train_blur_imgs), batch_size):
            augmentation = random.choice(augmentation_list)
            if augmentation == 0:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + batch_size],
                                                                            train_classification_mask[
                                                                            idx: idx + batch_size])],
                                                            fn=crop_sub_img_and_classification_fn)
            else:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + batch_size],
                                                                            train_classification_mask[
                                                                            idx: idx + batch_size])],
                                                            fn=crop_sub_img_and_classification_fn_aug)
            # print images_and_score.shape
            imlist, clist = images_and_score.transpose((1, 0, 2, 3, 4))
            clist = clist[:, :, :, 0]
            clist = np.expand_dims(clist, axis=3)

            err, l1, l2, l3, l4, _, outmap = sess.run([loss, loss1, loss2, loss3, loss4, train_op, out],
                                                      {net_regression.inputs: imlist,
                                                       classification_map: clist})
            total_loss += err
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f\n" % (epoch, n_epoch, time.time() - epoch_time,
                                                                        total_loss / n_iter)
        # only way to write to log file while running
        with open(checkpoint_dir + "/training_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        with open(checkpoint_dir + "/training_metrics.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, str(np.round(total_loss / n_iter, 8))])

        ## save model
        if epoch % 10 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                               save_dir=checkpoint_dir, var_list=a_vars, global_step=epoch, printable=False)


# updated synthetic training with python 3.8 tensorflow 2.0
def train_with_synthetic():
    checkpoint_dir = "test_checkpoint/{}".format(tl.global_flag['mode'])  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_config(checkpoint_dir + '/config', config)

    input_path = config.TRAIN.blur_path
    gt_path = config.TRAIN.gt_path
    if '.jpg' in tl.global_flag['image_extension']:
        train_blur_img_list = sorted(
            tl.files.load_file_list(path=input_path, regx='(out_of_focus|motion).*.(jpg|JPG)', printable=False))
    else:
        train_blur_img_list = sorted(
            tl.files.load_file_list(path=input_path, regx='(out_of_focus|motion).*.(png|PNG)', printable=False))
    train_mask_img_list = []

    for name in train_blur_img_list:
        if ".jpg" in name:
            train_mask_img_list.append(name.replace(".jpg", ".png"))
        else:
            train_mask_img_list.append(name.replace(".JPG", ".png"))

    ###Load Training Data ####
    train_blur_imgs = read_all_imgs(train_blur_img_list, path=input_path, n_threads=batch_size, mode='RGB')
    train_mask_imgs = read_all_imgs(train_mask_img_list, path=gt_path, n_threads=batch_size, mode='GRAY')

    index = 0
    train_classification_mask = []
    # print train_mask_imgs
    # img_n = 0
    for img in train_mask_imgs:
        tmp_class = img
        tmp_classification = np.concatenate((img, img, img), axis=2)

        tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
        tmp_class[np.where(tmp_classification[:, :, 0] == 100)] = 1  # motion blur
        tmp_class[np.where(tmp_classification[:, :, 0] == 200)] = 2  # defocus blur

        train_classification_mask.append(tmp_class)
        index = index + 1

    input_path2 = config.TRAIN.CUHK_blur_path

    if '.jpg' in tl.global_flag['image_extension']:
        ori_train_blur_img_list = sorted(
            tl.files.load_file_list(path=input_path2, regx='(out_of_focus|motion).*.(jpg|JPG)', printable=False))
    else:
        ori_train_blur_img_list = sorted(
            tl.files.load_file_list(path=input_path2, regx='(out_of_focus|motion).*.(png|PNG)', printable=False))
    ori_train_mask_img_list = []

    for name in ori_train_blur_img_list:

        if ".jpg" in name:
            ori_train_mask_img_list.append(name.replace(".jpg", ".png"))
        else:
            ori_train_mask_img_list.append(name.replace(".JPG", ".png"))

    # augmented dataset read
    gt_path2 = config.TRAIN.CUHK_gt_path
    origional_train_blur_imgs = read_all_imgs(ori_train_blur_img_list, path=input_path2, n_threads=batch_size,
                                              mode='RGB')
    origional_train_mask_imgs = read_all_imgs(ori_train_mask_img_list, path=gt_path2, n_threads=batch_size, mode='GRAY')

    ori_train_blur_imgs = []
    ori_train_mask_imgs = []

    # get only the chuk image list that is used for training 
    list_file = 'dataset/train_list.txt'
    with open(list_file) as f:
        lines = f.readlines()
    indexs = np.arange(0,len(origional_train_blur_imgs))
    testList = '\t'.join(lines)
    for idx in indexs:
        imagename = ori_train_blur_img_list[idx]
        if imagename in testList:
            ori_train_blur_imgs.append(origional_train_blur_imgs[idx])
            ori_train_mask_imgs.append(origional_train_mask_imgs[idx])

    # clean memeory
    origional_train_mask_imgs,origional_train_blur_imgs = None, None

    index = 0
    ori_train_classification_mask = []
    # img_n = 0
    for img in ori_train_mask_imgs:
        if index < 236:
            tmp_class = img
            tmp_classification = np.concatenate((img, img, img), axis=2)

            tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
            tmp_class[np.where(tmp_classification[:, :, 0] > 0)] = 1  # defocus blur
        else:
            tmp_class = img
            tmp_classification = np.concatenate((img, img, img), axis=2)
            tmp_class[np.where(tmp_classification[:, :, 0] == 0)] = 0  # sharp
            tmp_class[np.where(tmp_classification[:, :, 0] > 0)] = 2  # defocus blur

        ori_train_classification_mask.append(tmp_class)
        index = index + 1

    train_mask_imgs = train_classification_mask

    for i in range(10):
        train_blur_imgs = train_blur_imgs + ori_train_blur_imgs
        train_mask_imgs = train_mask_imgs + ori_train_classification_mask

    train_classification_mask = np.array(train_mask_imgs)
    train_blur_imgs = np.array(train_blur_imgs)

    # clean memeory
    train_mask_imgs,ori_train_classification_mask,ori_train_mask_imgs,ori_train_blur_imgs = None,None,None,None

    print("Number of training images " + str(len(train_blur_imgs)))

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [batch_size, h, w, 3], name='input_patches')
    classification_map = tf.compat.v1.placeholder('int32', [batch_size, h, w, 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, m1, m2, m3 = Decoder_Network_classification(input, n, f0, f0_1, f1_2, f2_3, 
            reuse=False,scope=scope2)

    ### DEFINE LOSS ###
    loss1 = tl.cost.cross_entropy(net_regression.outputs, tf.squeeze(classification_map), name='loss1')
    loss2 = tl.cost.cross_entropy(m1.outputs,
                                  tf.squeeze(tf.image.resize(classification_map, [int(h / 2), int(w / 2)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss2')
    loss3 = tl.cost.cross_entropy(m2.outputs,
                                  tf.squeeze(tf.image.resize(classification_map, [int(h / 4), int(w / 4)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss3')
    loss4 = tl.cost.cross_entropy(m3.outputs,
                                  tf.squeeze(tf.image.resize(classification_map, [int(h / 8), int(w / 8)],
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)),
                                  name='loss4')
    out = net_regression.outputs
    # output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
    loss = loss1 + loss2 + loss3 + loss4

    # as per kim et al paper
    with tf.compat.v1.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init * 0.1 * 0.1 * 0.1, trainable=False)
        lr_v2 = tf.Variable(lr_init * 0.1 * 0.1, trainable=False)

    ### DEFINE OPTIMIZER ###
    a_vars = tl.layers.get_variables_with_name('Unified', False, True)  # Unified
    var_list1 = tl.layers.get_variables_with_name('VGG', True, True)  # ?
    var_list2 = tl.layers.get_variables_with_name('UNet', True, True)  # ?
    opt1 = tf.optimizers.Adam(learning_rate=lr_v)
    opt2 = tf.optimizers.Adam(learning_rate=lr_v2)
    grads = tf.gradients(ys=loss, xs=var_list1 + var_list2, unconnected_gradients='zero')
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op = tf.group(train_op1, train_op2)

    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    # uncomment if on a gpu machine
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    
        except RuntimeError as e:
            print(e)
    sess = tf.compat.v1.Session(config=configTf)

    print("initializing global variable...")
    sess.run(tf.compat.v1.global_variables_initializer())
    print("initializing global variable...DONE")

    ### LOAD VGG else load previous weights from starting from###
    if tl.global_flag['start_from'] != 0:
        print("loading initial checkpoint")
        tl.files.load_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                           save_dir=checkpoint_dir, var_list=a_vars, is_latest=True, printable=True)
    else:
        ### initial checkpoint ###
        checkpoint_dir2 = "test_checkpoint/PG_CUHK/"
        tl.files.load_ckpt(sess=sess, mode_name='SA_net_PG_CUHK.ckpt', save_dir=checkpoint_dir2, var_list=a_vars,
                           is_latest=True)

    global_step = 0
    new_lr_decay = 1

    ### START TRAINING ###
    augmentation_list = [0, 1]

    # initialize the csv metrics output
    with open(checkpoint_dir + "/training_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'total_error'])

    for epoch in range(tl.global_flag['start_from'], n_epoch + 1):
        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            sess.run(tf.compat.v1.assign(lr_v, lr_v * lr_decay))
            sess.run(tf.compat.v1.assign(lr_v2, lr_v2 * lr_decay))
            log = " ** new learning rate for Encoder %f\n" % (sess.run(lr_v))
            with open(checkpoint_dir + "/training_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
            log = " ** new learning rate for Decoder %f\n" % (sess.run(lr_v2))
            with open(checkpoint_dir + "/training_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
        elif epoch == tl.global_flag['start_from']:
            log = " ** init lr for Decoder: %f decay_every_init: %d, lr_decay: %f\n" % (sess.run(lr_v2), decay_every,
                                                                                        lr_decay)
            with open(checkpoint_dir + "/training_metrics.log", "a") as f:
                # perform file operations
                f.write(log)
            log = " ** init lr for Encoder: %f decay_every_init: %d, lr_decay: %f\n" % (sess.run(lr_v), decay_every,
                                                                                        lr_decay)
            # print(log)
            with open(checkpoint_dir + "/training_metrics.log", "a") as f:
                # perform file operations
                f.write(log)

        epoch_time = time.time()
        total_loss, n_iter = 0, 0

        # data shuffle***
        train_blur_imgs, train_classification_mask = unison_shuffled_copies(train_blur_imgs, train_classification_mask)

        for idx in range(0, len(train_blur_imgs), batch_size):
            augmentation = random.choice(augmentation_list)
            if augmentation == 0:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + batch_size],
                                                                            train_classification_mask[
                                                                            idx: idx + batch_size])],
                                                            fn=crop_sub_img_and_classification_fn)
            elif augmentation == 1:
                images_and_score = tl.prepro.threading_data([_ for _ in zip(train_blur_imgs[idx: idx + batch_size],
                                                                            train_classification_mask[
                                                                            idx: idx + batch_size])],
                                                            fn=crop_sub_img_and_classification_fn_aug)
            # print images_and_score.shape
            imlist, clist = images_and_score.transpose((1, 0, 2, 3, 4))
            clist = clist[:, :, :, 0]
            clist = np.expand_dims(clist, axis=3)

            err, l1, l2, l3, l4, _, outmap = sess.run([loss, loss1, loss2, loss3, loss4, train_op, out],
                                                      {net_regression.inputs: imlist,
                                                       classification_map: clist})
            total_loss += err
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %.8f\n" % (epoch, n_epoch, time.time() - epoch_time,
                                                                        total_loss / n_iter)
        # only way to write to log file while running
        with open(checkpoint_dir + "/training_metrics.log", "a") as f:
            # perform file operations
            f.write(log)
        with open(checkpoint_dir + "/training_metrics.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, str(np.round(total_loss / n_iter, 8))])

        ## save model
        if epoch % 10 == 0:
            tl.files.save_ckpt(sess=sess, mode_name='SA_net_{}.ckpt'.format(tl.global_flag['mode']),
                               save_dir=checkpoint_dir, var_list=a_vars, global_step=epoch, printable=False)
