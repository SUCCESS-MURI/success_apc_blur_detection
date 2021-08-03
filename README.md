## NEED TO UPDATE THIS 
previous project's readme 

# MURI Deep Blur Detection and Classification
# Update images 
%![input2](./input/out_of_focus0607.jpg) ![output2](./output/out_of_focus0607.png)

This repository contains the code and dataset locations for the muri integration project and the origional datasets used by the origional project https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.

--------------------------

## Prerequisites (tested)
- Ubuntu 20.04
- Tensorflow-gpu 2.2.0
- Tensorlayer 2.2.3
- OpenCV2
- Listed in the document Packages_Requirements
- Using Ananconda Virtual Enviornment

install anaconda and packages from the commands found in the doc XXXXX

## Model Pre-Trained Weights Training Details
- ## CHUK Dataset
- We used [CUHK blur detection dataset](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html) for training our network and generating our synthetic dataset
- Train and test set lists are uploaded in 'dataset' folder
- Need to modify some options and paths in 'main.py' and 'config.py' for training
- ## Synthetic Dataset
- download [synthetic train set](https://drive.google.com/file/d/1QUygL2nalHldcJMwFJPfPFWokMoIbI9L/view?usp=sharing)(337MB) and [synthetic test set](https://drive.google.com/file/d/1-lV3CS_6rI_by6StkGQYsdn0SeOxwepu/view?usp=sharing)(11.5MB) from google drive
- Note that sharp pixels, motion-blurred pixels, and defocus-blurred pixels in GT blur maps are labeled as 0, 100, and 200, respectively, in the [0,255] range.

## Test Details
- download [model weights](https://drive.google.com/file/d/11FBVmAIfeHDHpOjLXewzpA2lgcOOqo2_/view?usp=sharing) from google drive and save the model into 'model' folder.
- specify a path of input folder in 'main.py' at line #39
- run 'main.py'

```bash
python main.py
```

## Sample output 
in the folder sample_blur_output there are 3 folders:

gt_images - rgb ground truth images for blur detection (pink-brightness, red - darkness, green - focus, blue - motion)

output_images - rgb images for that were outputed by blur detection (pink-brightness, red - darkness, green - focus, blue - motion) 

raw_npy_output - softmax blur output (size (256,256,5)) the indexs indicate the following: 0: no blur 1: motion blur 2: focus blur 3: darkness blur 4: brightness blur

## License ##
NEED to Update for MURI/CMU

This software is being made available under the terms in the [LICENSE](LICENSE) file.
Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

