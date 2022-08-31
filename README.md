# Blur Detection and Classification with Exposure Blur Labels

This repository contains the training and test code for running our modifed Kim et al. blur detection model with the additional 2 classification labels for over-and underexposure. This repo was updated to run with the updated tensorflow 2.0 and python 3.8 version [repo](https://github.com/SUCCESS-MURI/success_apc_blur_detection/releases/tag/v2.0.0)

## Kim et al. Train and Testing Details from [Original Repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git)
## Prerequisites 
- Please clone the release branch [Deep_Blur_Detection_and_Classification_w_Tensorflow2_Python3.8](https://github.com/SUCCESS-MURI/success_apc_blur_detection/releases/tag/v2.0.0) which is the original model branch updated to use tensorflow 2.0 and python 3.8. 
- Follow all instructions for env and libraries in workplace_setup folder and from [Deep_Blur_Detection_and_Classification_w_Tensorflow2_Python3.8](https://github.com/SUCCESS-MURI/success_apc_blur_detection/releases/tag/v2.0.0)
- Our [exposure blur detection datasets](https://bridge.apt.ri.cmu.edu/exposure/real_synthetic_dataset) was used for training and testing our network (This download is 4.3 GBs)
- Train and test set details are in the readme in the dataset download

## Training Details
- You will need to modify some options and paths in 'new_main.py' and 'config.py' for training
- Same input/config arguments as in [Deep_Blur_Detection_and_Classification_w_Tensorflow2_Python3.8](https://github.com/SUCCESS-MURI/success_apc_blur_detection/releases/tag/v2.0.0) 
- Please modify config.py with updated dataset locations for training where it is located in local directory
- new_main.py has the following new input options (old inputs from [updated repo](https://github.com/SUCCESS-MURI/success_apc_blur_detection/releases/tag/v2.0.0) except for --prev_weights are still valid):
  - --exposure (true,false default false): running exposure classification or not

Run the following example command after all of your input parameters are set in config.py 
```bash
python new_main.py --is_train true --exposure true
```

## Test Details
- download [model weights](https://bridge.apt.ri.cmu.edu/exposure/real_synthetic_weights) and save the model into 'model' folder.
- Can use the converted [.npy weights](https://bridge.apt.ri.cmu.edu/exposure/npy_kim_weights) from the [original repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git).
- new_main.py has the following new input options for testing (old inputs from [updated repo](https://github.com/SUCCESS-MURI/success_apc_blur_detection/releases/tag/v2.0.0) are still valid):
  - --exposure (true,false default false): running exposure classification or not
  - --exposure_dataset (true, false, default false): use this boolean if you want to test origional kim et al. model with exposure blur images

```bash
python main.py --exposure true
```

## Running Real Time with ROS
ROSService for real time implementation in ros 

### Normal Blur DEtection and Classifcation
Sevice message called BlurOutput.srv
Input
- sensor_msgs/Image bgr : bgr image input 
- string save_name : file name including file extension for blur output image
Output
- bool success : service success of failure boolean
- std_msgs/Int32 label : argmax label for most type of blur used
- string msg : error message if service failed

Run the launch file which uses blur_detection_ros_main.py parameters include
- model_checkpoint: location of the weights .ckpt you want to use 
- height: input image height
- width: input image width

Run the launch file
```
roslaunch success_apc_blur_detection blur_detection.launch
```

The following service message is called

```
bash rospy.ServiceProxy('/success_apc_blur_detection/blurdetection', BlurOutput)
```
 Follow the ROSService instructions for full description on how to send and receive messages
 

### Using a salencey mask for local blur detection
Sevice message called BlurMaskOutput.srv
Input
- sensor_msgs/Image bgr : bgr image input 
- sensor_msgs/Image mask : mask image input 
- string save_name : file name including file extension for blur output image
Output
- bool success : service success of failure boolean
- std_msgs/Int32 label : argmax label for most type of blur used
- string msg : error message if service failed

Another launch file uses blur_detection_ros_main_saliency.py parameters include
- model_checkpoint: location of the weights .ckpt you want to use 
- height: input image height
- width: input image width

Run the launch file
```
roslaunch success_apc_blur_detection blur_detection_w_saliency.launch
```

Following service message is called

```bash
rospy.ServiceProxy('/success_apc_blur_detection/blurdetection', BlurMaskOutput)
```

## Avaliable Datasets and Weights 
Real and Syntheic Exposure Training and Testing Datasets [here](https://bridge.apt.ri.cmu.edu/exposure/real_synthetic_dataset)

IPS Real Exposure Training and Testing Datasets [here](https://bridge.apt.ri.cmu.edu/exposure/ips_real_dataset)

Real and Syntheic Exposure Weights [here](https://bridge.apt.ri.cmu.edu/exposure/real_synthetic_weights)

IPS Real Exposure Weights [here](https://bridge.apt.ri.cmu.edu/exposure/ips_real_weights)

## License ##
MIT License

Copyright (c) 2022 Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

