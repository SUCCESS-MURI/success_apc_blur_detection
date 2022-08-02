# coding=utf-8
import argparse
from test_updated import test_kimetal_origional_model_on_exposure_dataset, exposure_test, exposure_test_with_no_gt
from test import test, test_with_no_gt
import tensorlayer as tl
from train_updated import exposure_training
from train import train_with_synthetic, train_with_CHUK

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
    parser.add_argument('--is_synthetic', type=str, default='false', help='whether synthetic training or not')
    parser.add_argument('--exposure', type=str, default='false', help='whether using exposure classifcation or not')
    parser.add_argument('--exposure_dataset', type=str, default='false', help='whether using exposure dataset or not. '
                                    'Only needed if using kim et al testing with exposure labeled images')
    parser.add_argument('--start_from', type=int, default='0', help='start from')
    parser.add_argument('--gpu', type=str, default='false', help='whether synthetic training or not')
    parser.add_argument('--uncertainty_label', type=str, default='false', help='whether testing result should have an '
                                                                               'uncertanity label. default is false')
    parser.add_argument('--output_levels', type=str, default='false', help='output each of the levels for testing')
    parser.add_argument('--test_wo_gt', type=str, default='false', help='testing images without gt')
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['start_from'] = int(args.start_from)
    tl.global_flag['uncertainty_label'] = t_or_f(args.uncertainty_label)
    tl.global_flag['gpu'] = t_or_f(args.gpu)
    tl.global_flag['exposure'] = t_or_f(args.exposure)
    tl.global_flag['is_synthetic'] = t_or_f(args.is_synthetic)
    tl.global_flag['output_levels'] = t_or_f(args.output_levels)
    tl.global_flag['test_wo_gt'] = t_or_f(args.test_wo_gt)
    tl.global_flag['exposure_dataset'] = t_or_f(args.exposure_dataset)

    # training scenarios
    if tl.global_flag['is_train']:
        if tl.global_flag['is_synthetic']:
            train_with_synthetic()
        elif tl.global_flag['exposure']:
            exposure_training()
        else:
            train_with_CHUK()
    else: # testing scenarios
        if tl.global_flag['exposure']:
            if tl.global_flag['test_wo_gt']:
                exposure_test_with_no_gt()
            else:
                exposure_test()
        elif tl.global_flag['exposure_dataset']:
            test_kimetal_origional_model_on_exposure_dataset()
        else:
            if tl.global_flag['test_wo_gt']:
                test_with_no_gt()
            else:
                test()