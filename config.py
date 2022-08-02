from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()

## Adam
config.TRAIN.batch_size =10
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 6000
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 6000

config.TRAIN.blur_path = '/home/mary/code/local_success_dataset/fetch_images/output_blur_data_06_20_2022/Blur_Detection_Input/Testing/images/'
config.TRAIN.gt_path = '/home/mary/code/local_success_dataset/fetch_images/output_blur_data_06_20_2022/Blur_Detection_Input/Testing/gt/'

config.TEST.blur_path = '/Dataset/BlurDetection/CUHK/test/imgs/'
config.TEST.gt_path = '/Dataset/BlurDetection/CUHK/test/GT/'

# for syntheic training needs chuk path
config.TRAIN.CUHK_blur_path = '/Dataset/BlurDetection/CUHK/test/imgs/'
config.TRAIN.CUHK_gt_path = '/Dataset/BlurDetection/CUHK/test/gt/'

## train image size
config.TRAIN.height = 256#192->144 ->288
config.TRAIN.width = 256 #->288

config.TEST.height = 480
config.TEST.width = 640

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
