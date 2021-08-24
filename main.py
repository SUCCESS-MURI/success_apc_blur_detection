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
import sys
from train import *
from test import *

# https://intellipaat.com/community/4920/parsing-boolean-values-with-argparse
def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        pass  # Here you can write error condition.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='PG_CUHK', help='model name')
    parser.add_argument('--is_train', type=str , default='false', help='whether train or not')
    parser.add_argument('--is_synthetic', type=str, default='false', help='whether synthetic train or not')
    parser.add_argument('--is_ssc_dataset', type=str, default='false', help='whether ssc dataset train or not')
    parser.add_argument('--index', type=int, default='0', help='index range 50')
    parser.add_argument('--start_from', type=int, default='0', help='start from')
    parser.add_argument('--uncertainty_label', type=str, default='false', help='whether testing result should have an uncertanity label. default is false')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['is_synthetic'] = t_or_f(args.is_synthetic)
    tl.global_flag['is_ssc_dataset'] = t_or_f(args.is_ssc_dataset)
    tl.global_flag['start_from'] = int(args.start_from)
    tl.global_flag['uncertainty_label'] = t_or_f(args.uncertainty_label)

    if tl.global_flag['is_train']:
        if tl.global_flag['is_synthetic']:
            train_with_synthetic() # train with the CUHK dataset first and then finetune with the synthetic dataset
        elif tl.global_flag['is_ssc_dataset']:
            train_with_ssc_dataset()
            # BACKBURNNNER right now this has alot to work out with not alot of gain. currently i think i am the most optimized i can be for now.
            # UPDATED_train_with_ssc_dataset()
        else:
            train_with_chuk_updated_dataset()
    else:
        #blurmap_3classes(args.index) #pg test
        #blurmap_3classes_using_numpy_pretrainied_weights(args.index)
        testData_return_error()

