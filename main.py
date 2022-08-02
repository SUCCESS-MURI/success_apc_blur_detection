# coding=utf-8
import argparse
from train import *
from test import test, test_with_no_gt

# https://intellipaat.com/community/4920/parsing-boolean-values-with-argparse
def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        pass

# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # from kim et al.
    parser.add_argument('--mode', type=str, default='PG_CUHK', help='model name')
    parser.add_argument('--is_train', type=str , default='false', help='whether train or not')
    parser.add_argument('--is_synthetic', type=str, default='false', help='synthetic training')
    parser.add_argument('--output_levels', type=str, default='false', help='output each of the levels for testing')
    parser.add_argument('--start_from', type=int, default='0', help='if training was paused start from an index and '
                                                                    'init weights')
    parser.add_argument('--test_wo_gt', type=str, default='false', help='testing images without gt')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['is_synthetic'] = t_or_f(args.is_synthetic)
    tl.global_flag['start_from'] = int(args.start_from)
    tl.global_flag['output_levels'] = int(args.output_levels)
    tl.global_flag['test_wo_gt'] = int(args.test_wo_gt)

    if tl.global_flag['is_train']:
        if not tl.global_flag['is_synthetic']:
            train_with_CHUK()
        else:
            train_with_synthetic()
    else:
        if tl.global_flag['test_wo_gt']:
            test_with_no_gt()
        else:
            # loads the original weights in the format of the .npy file
            test()




