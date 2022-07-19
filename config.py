from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()
config.VALIDATION = edict()
config.SENSITVITY = edict()

## Adam
config.TRAIN.batch_size = 10
config.TRAIN.lr_init = 1e-5
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epochs = 1000
config.TRAIN.valid_every = 50

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 500
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = 6000

config.TRAIN.blur_path = '/home/mary/code/local_success_dataset/fetch_images/output_blur_data_06_20_2022/Blur_Detection_Input/Testing/images/'
config.TRAIN.gt_path = '/home/mary/code/local_success_dataset/fetch_images/output_blur_data_06_20_2022/Blur_Detection_Input/Testing/gt/'

## train set location
config.TRAIN.muri_blur_path = '/home/mary/code/local_success_dataset/muri_dataset/Testing_2/images/'
config.TRAIN.muri_gt_path = '/home/mary/code/local_success_dataset/muri_dataset/Testing_2/gt/'

config.VALIDATION.muri_blur_path = '/home/mary/code/local_success_dataset/muri_dataset/Validation_2/images/'
config.VALIDATION.muri_gt_path = '/home/mary/code/local_success_dataset/muri_dataset/Validation_2/gt/'

config.TEST.muri_blur_path = '/home/mary/code/local_success_dataset/muri_dataset_01_12_2022/Testing/images/'
config.TEST.muri_gt_path = '/home/mary/code/local_success_dataset/muri_dataset_01_12_2022/Testing/gt/'

config.TEST.blur_path = '/home/mary/code/local_success_dataset/datasets_iros/IROS_Dataset/02_14_2022_run2/Testing/images'
config.TEST.gt_path = '/home/mary/code/local_success_dataset/datasets_iros/IROS_Dataset/02_14_2022_run2/Testing/gt'

config.VALIDATION.blur_path = '/home/mary/code/local_success_dataset/test/Validation/images'
config.VALIDATION.gt_path = '/home/mary/code/local_success_dataset/test/Validation/gt'

config.SENSITVITY.muri_blur_path = '/home/mary/code/local_success_dataset/muri_dataset_01_12_2022/Sensitivity/images' \
                                   '/halfsize/'
config.SENSITVITY.muri_gt_path = '/home/mary/code/local_success_dataset/muri_dataset_01_12_2022/Sensitivity/gt' \
                                 '/halfsize/'
config.TEST.real_blur_path = '/home/mary/code/local_success_dataset/IROS_Dataset/examples_for_autoexposure'#'/home/mary/Desktop/testing/'

## train image size
config.TRAIN.height = 256
config.TRAIN.width = 256

config.TEST.height = 480
config.TEST.width = 640

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
