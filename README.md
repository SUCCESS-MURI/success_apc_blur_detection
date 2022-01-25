# Blur Detection and Classifcation 

This repository contains the training and test code for running our blur detection model based on the Kim et al. model. This repo was updated to run with tensorflow 2.0 and python 3.8

--------------------------
## Prerequisites (tested)
- Ubuntu 20.04
- Python 3.8
- Tensorflow-gpu 2.2.0
- Tensorlayer 2.2.3
- OpenCV2
- Listed in the document Packages_Requirements
- Using Ananconda Virtual Enviornment

### Note with Tensorflow update
read the tensorflow_issues_and_solutions document for code solutions for errors that might arise with updating tensorflow

## Kim et al. Train and Testing Details from [Original Repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git)
## Train Details
- We used [CUHK blur detection dataset](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html) for training our network and generating our synthetic dataset
- Train and test set lists are uploaded in 'dataset' folder in [repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git)
- Need to modify some options and paths in 'main.py' and 'config.py' for training

```bash
python main.py --is_train true
```

## Test Details
- download [model weights](https://drive.google.com/file/d/11FBVmAIfeHDHpOjLXewzpA2lgcOOqo2_/view?usp=sharing) from google drive and save the model into 'model' folder.
- specify a path of input folder in 'main.py' at line #39
- run 'main.py'

```bash
python main.py --is_train false
```
- Note there are 2 ways of running the test weights. Can use the converted .npy weights on XXX from the [original repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git) or you can run training and use .cktp weights generated from this code. 
- 
## Synthetic Dataset
- download [synthetic train set](https://drive.google.com/file/d/1QUygL2nalHldcJMwFJPfPFWokMoIbI9L/view?usp=sharing)(337MB) and [synthetic test set](https://drive.google.com/file/d/1-lV3CS_6rI_by6StkGQYsdn0SeOxwepu/view?usp=sharing)(11.5MB) from google drive
- Note that sharp pixels, motion-blurred pixels, and defocus-blurred pixels in GT blur maps are labeled as 0, 100, and 200, respectively, in the [0,255] range.

```bash
python main.py --is_train true --is_synthetic true
```
## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.
Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

