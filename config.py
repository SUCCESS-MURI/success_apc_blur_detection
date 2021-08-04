from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()
config.VALIDATION = edict()

## Adam
config.TRAIN.batch_size = 10
config.TRAIN.lr_init = 1e-5
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1000
config.TRAIN.valid_every = 50 # TODO change to 100

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 500
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = 6000

## train set location
config.TRAIN.muri_blur_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Training/images/'
config.TRAIN.muri_gt_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Training/gt/'

config.VALIDATION.muri_blur_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Validation/images/'
config.VALIDATION.muri_gt_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Validation/gt/'

config.TEST.muri_blur_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Testing/images/'
config.TEST.muri_gt_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Testing/gt/'

## train image size
config.TRAIN.height = 256
config.TRAIN.width = 256

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
