import os

import numpy as np
from PIL import Image

sharp_gt_path = '/home/mary/code/local_success_dataset/CHUK_Dataset/08_25_2021/Training/gt/'  # ./input'
test_sharp_gt_list = os.listdir(sharp_gt_path)
test_sharp_gt_list.sort()

count_total = 0

for i in range(len(test_sharp_gt_list)):
    gt = test_sharp_gt_list[i]

    sharp_gt = os.path.join(sharp_gt_path, gt)
    sharp_gt_image = Image.open(sharp_gt)
    sharp_gt_image.load()

    sharp_gt_image = np.asarray(sharp_gt_image, dtype="int64")
    if len(sharp_gt_image[sharp_gt_image == 0]) != 0 and len(sharp_gt_image[sharp_gt_image == 64]) != 0 and len(sharp_gt_image[sharp_gt_image == 128]) != 0 and len(sharp_gt_image[sharp_gt_image == 192]) != 0 and len(sharp_gt_image[sharp_gt_image == 255]) != 0:
        count_total += 1
    else:
        continue

print(count_total)