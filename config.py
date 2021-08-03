from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()
config.VALIDATION = edict()

## Adam
config.TRAIN.batch_size = 10
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1000

config.TRAIN.valid_every = 10 # TODO change to 100

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 20
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = 6000

## train set location
config.TRAIN.CUHK_blur_path = '/home/mary/code/local_success_dataset/CUHK_Split_Dataset/Training/image/'
config.TRAIN.CUHK_gt_path = '/home/mary/code/local_success_dataset/CUHK_Split_Dataset/Training/gt/'
# config.TRAIN.synthetic_blur_path = '/home/mary/code/local_success_dataset/BlurTrainSyntheic/imgs/'
# config.TRAIN.synthetic_gt_path = '/home/mary/code/local_success_dataset/BlurTrainSyntheic/GT/'
config.TRAIN.synthetic_blur_path = '/home/mary/code/local_success_dataset/CUHK_Split_Dataset/Training/image/'
config.TRAIN.synthetic_gt_path = '/home/mary/code/local_success_dataset/CUHK_Split_Dataset/Training/gt/'

config.TRAIN.ssc_blur_path = '/home/mary/code/local_success_dataset/BlurDetectionDataset/Training_Resized_small/image/'
config.TRAIN.ssc_gt_path = '/home/mary/code/local_success_dataset/BlurDetectionDataset/Training_Resized_small/gt/'

config.TEST.ssc_blur_path = '/home/mary/code/local_success_dataset/BlurDetectionDataset/Testing_Resized_2/image/'
config.TEST.ssc_gt_path = '/home/mary/code/local_success_dataset/BlurDetectionDataset/Testing_Resized_2/gt/'

config.TRAIN.muri_blur_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Training/images/'
config.TRAIN.muri_gt_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Training/gt/'

config.VALIDATION.muri_blur_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Validation/images/'
config.VALIDATION.muri_gt_path = '/home/mary/code/local_success_dataset/muri/muri_dataset/Validation/gt/'

## train image size
config.TRAIN.height = 256
config.TRAIN.width = 256

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
