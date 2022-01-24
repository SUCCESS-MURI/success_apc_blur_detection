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

# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # from kim et al.
    parser.add_argument('--mode', type=str, default='PG_CUHK', help='model name')
    parser.add_argument('--is_train', type=str , default='false', help='whether train or not')
    parser.add_argument('--is_synthetic', type=str, default='false', help='synthetic training')
    parser.add_argument('--start_from', type=int, default='0', help='if training was paused start from an index and '
                                                                    'init weights')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['is_synthetic'] = t_or_f(args.is_syntheic)
    tl.global_flag['start_from'] = int(args.start_from)

    if tl.global_flag['is_train']:
        if not tl.global_flag['is_synthetic']:
            train_with_CHUK()
        else:
            train_with_synthetic()
    else:
        # loads the original weights in the format of the .npy file
        test()




