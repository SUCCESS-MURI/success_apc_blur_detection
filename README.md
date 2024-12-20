# deep_blur_detection_and_classification


%![input2](./input/out_of_focus0607.jpg) ![output2](./output/out_of_focus0607.png)

This repository contains a test code and sythetic dataset, which consists of scenes including motion and defocus blurs together in each scene.

--------------------------

## Prerequisites (tested)
- Ubuntu 20.04
- Tensorflow-gpu 2.2.0
- Tensorlayer 2.2.3
- OpenCV2
- Listed in the document Packages_Requirements
- Using Ananconda Virtual Enviornment

## Train Details
- We used [CUHK blur detection dataset](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html) for training our network and generating our synthetic dataset
- Train and test set lists are uploaded in 'dataset' folder
- Need to modify some options and paths in 'main.py' and 'config.py' for training

## Test Details
- download [model weights](https://drive.google.com/file/d/11FBVmAIfeHDHpOjLXewzpA2lgcOOqo2_/view?usp=sharing) from google drive and save the model into 'model' folder.
- specify a path of input folder in 'main.py' at line #39
- run 'main.py'

```bash
python main.py
```
## Synthetic Dataset
- download [synthetic train set](https://drive.google.com/file/d/1QUygL2nalHldcJMwFJPfPFWokMoIbI9L/view?usp=sharing)(337MB) and [synthetic test set](https://drive.google.com/file/d/1-lV3CS_6rI_by6StkGQYsdn0SeOxwepu/view?usp=sharing)(11.5MB) from google drive
- Note that sharp pixels, motion-blurred pixels, and defocus-blurred pixels in GT blur maps are labeled as 0, 100, and 200, respectively, in the [0,255] range.

## Sample output 
in the folder sample_blur_output there are 3 folders:

gt_images - rgb ground truth images for blur detection (pink-brightness, red - darkness, green - focus, blue - motion)

output_images - rgb images for that were outputed by blur detection (pink-brightness, red - darkness, green - focus, blue - motion) 

raw_npy_output - softmax blur output (size (256,256,5)) the indexs indicate the following: 0: no blur 1: motion blur 2: focus blur 3: darkness blur 4: brightness blur

## License ##

This software is being made available under the terms in the [LICENSE](LICENSE) file.

