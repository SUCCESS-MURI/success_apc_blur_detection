# Blur Detection and Classification 

This repository contains the training and test code for running our blur detection model based on the Kim et al. model. This repo was updated to run with tensorflow 2.0 and python 3.8

--------------------------
## Prerequisites (tested)
- Ubuntu 20.04
- Python 3.8
- Tensorflow-gpu 2.2.0
- Tensorlayer 2.2.3
- OpenCV2
- Other packages needed are listed in the document Packages_Requirements.txt
- Using Ananconda Virtual Enviornment

### Note with Tensorflow update
read the tensorflow_issues_and_solutions document for code solutions for errors that might arise with updating tensorflow

## Kim et al. Train and Testing Details from [Original Repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git)
## Train Details
- We used [CUHK blur detection dataset](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html) for training our network and generating our synthetic dataset
- Train and test set lists are uploaded in 'dataset' folder
- Need to modify some options and paths in 'main.py' and 'config.py' for training
- Please modify config.py with updated dataset locations for training where it is located in local directory
- main.py has the following input options:
  - --is_train (true,false default false): running training or testing
  - --mode (any name default PG_CUHK): name of training you want to use
  - --is_synthetic (true,false default false): running training or testing for syntheic kim et al images 
  - --start_from (int default 0): starting training from certain point - start from last frequency that the weights were saved at
  - --image_extension (.png or .jpg default .jpg): image extension for images input
  - --gpu (true,false default false): using gpu to run
- options for config.py include 
  - config.TRAIN.batch_size: batch size for training
  - config.TRAIN.lr_init: inital training learning rate
  - config.TRAIN.n_epoch: number of epochs 
  - config.TRAIN.lr_decay: decay rate
  - config.TRAIN.decay_every: decay after # of epochs 
  - config.TRAIN.blur_path: location of training images 
  - config.TRAIN.gt_path: location of ground truth images
  - config.TRAIN.height: height to use for images (will be cropped for training)
  - config.TRAIN.width: width to use for images (will be cropped for training)

```bash
python main.py --is_train true
```

## Synthetic Dataset
- download [synthetic train set](https://drive.google.com/file/d/1QUygL2nalHldcJMwFJPfPFWokMoIbI9L/view?usp=sharing)(337MB) and [synthetic test set](https://drive.google.com/file/d/1-lV3CS_6rI_by6StkGQYsdn0SeOxwepu/view?usp=sharing)(11.5MB) from google drive
- Note that sharp pixels, motion-blurred pixels, and defocus-blurred pixels in GT blur maps are labeled as 0, 100, and 200, respectively, in the [0,255] range.
- Same inputs as above but now use --is_synthetic true and new input 
--prev_weights (name of mode): mode used for chuk training. need to perform chuk training before doing syntheic training as per kim et al. 

```bash
python main.py --is_train true --is_synthetic true --prev_weights PG_CUHK
```

## Test Details
- Download the converted [.npy weights](https://bridge.apt.ri.cmu.edu/exposure/npy_kim_weights). These weights were converted to .npy from the [original repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git).
- main.py has the following input testing options:
  - --mode (any name default PG_CUHK): name of weights saved from training (same as mode from above) you want to use
  - --output_levels (true/false default false): outputs the individual level images from each decoder layer in network
  - --test_wo_gt (true/false default false): version of testing where we do not have ground truth. Useful for testing recorded images that do not have a ground truth assigned to them
  - --image_extension (.png or .jpg default .jpg): image extension for images input
- options for config.py include
  - config.TEST.blur_path: location of testing images 
  - config.TEST.gt_path: location of ground truth images
  - config.TEST.height: height to use for images (will be cropped for testing)
  - config.TEST.width: width to use for images (will be cropped for testing)

```bash
python main.py
```

## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.
Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

