# coding=utf-8
import argparse
import tensorlayer as tl
from train import train_with_muri_dataset
from test import test_with_muri_dataset, test_with_muri_dataset_origonal_model, test_with_muri_dataset_blur_noblur


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
    parser.add_argument('--start_from', type=int, default='0', help='start from')
    parser.add_argument('--uncertainty_label', type=str, default='false', help='whether testing result should have an uncertanity label. default is false')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['start_from'] = int(args.start_from)
    tl.global_flag['uncertainty_label'] = t_or_f(args.uncertainty_label)

    if tl.global_flag['is_train']:
        train_with_muri_dataset()
    else:
        test_with_muri_dataset()
        #test_with_muri_dataset_blur_noblur()
        #test_with_muri_dataset_origonal_model()

