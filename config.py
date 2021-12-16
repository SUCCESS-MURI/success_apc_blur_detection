from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()
config.VALIDATION = edict()

## Adam
config.TRAIN.batch_size = 10
config.TRAIN.lr_init = 1e-4

## epochs and validation
config.TRAIN.n_epochs = 2000
config.TRAIN.valid_every = 10

# learning rate
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = 6000

## train set location
config.TRAIN.blur_path = '/home/mary/code/local_success_dataset/CHUK_Dataset/08_20_2021/smaller_dataset/images/'
config.TRAIN.gt_path = '/home/mary/code/local_success_dataset/CHUK_Dataset/08_20_2021/smaller_dataset/gt/'

# config.TEST.ssc_blur_path = '/home/mary/code/local_success_dataset/CHUK_Dataset/08_25_2021/Testing_bd/images/'
# config.TEST.ssc_gt_path = '/home/mary/code/local_success_dataset/CHUK_Dataset/08_25_2021/Testing_bd/gt/'

## train image size
config.TRAIN.height = 256
config.TRAIN.width = 256

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
