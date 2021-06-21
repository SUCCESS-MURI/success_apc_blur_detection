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
    parser.add_argument('--index', type=int, default='0', help='index range 50')
    parser.add_argument('--start_from', type=int, default='0', help='start from')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['is_synthetic'] = t_or_f(args.is_synthetic)
    tl.global_flag['start_from'] = int(args.start_from)

    if tl.global_flag['is_train']:
        if tl.global_flag['is_synthetic']:
            train_with_synthetic() # train with the CUHK dataset first and then finetune with the synthetic dataset
        else:
            train_with_CUHK()
    else:
        blurmap_3classes(args.index) #pg test

