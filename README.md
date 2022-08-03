# Blur Detection and Classification with Exposure Blur Labels

This repository contains the training and test code for running Kim et al. blur detection model with the additional 2 classification labels for over-and underexposure. This repo was updated to run with tensorflow 2.0 and python 3.8

## Kim et al. Train and Testing Details from [Original Repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git)
## Prerequisites 
- Please clone the release branch [Deep_Blur_Detection_and_Classification_w_Tensorflow2_Python3.8](https://github.com/SUCCESS-MURI/success_apc_blur_detection.git) which is the original model branch updated to use tensorflow 2.0 and python 3.8. 
- Follow all instructions for env and libraries in workplace_setup folder and from [Deep_Blur_Detection_and_Classification_w_Tensorflow2_Python3.8](https://github.com/SUCCESS-MURI/success_apc_blur_detection.git)
- Our [exposure blur detection datasets](XXXX) was used for training and testing our network and generating our synthetic dataset
- Train and test set details are [here](XXXX). 

## Training Details
- You will need to modify some options and paths in 'new_main.py' and 'config.py' for training
- Same input/config arguments as in [Deep_Blur_Detection_and_Classification_w_Tensorflow2_Python3.8](https://github.com/SUCCESS-MURI/success_apc_blur_detection.git) 
- Please modify config.py with updated dataset locations for training where it is located in local directory
- new_main.py has the following new input options (old inputs from [updated repo](https://github.com/SUCCESS-MURI/success_apc_blur_detection.git) are still valid):
  - --exposure (true,false default false): running exposure classification or not
  - --gpu (true,false deafult false): using gpus for running

Run the following example command after all of your input parameters are set in config.py 
```bash
python new_main.py --is_train true --exposure true
```

## Test Details
- download [model weights](https://drive.google.com/file/d/11FBVmAIfeHDHpOjLXewzpA2lgcOOqo2_/view?usp=sharing) from google drive and save the model into 'model' folder.
- Can use the converted .npy weights on XXX from the [original repo](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification.git).
- new_main.py has the following new input options for testing (old inputs from [updated repo](https://github.com/SUCCESS-MURI/success_apc_blur_detection.git) are still valid):
  - --exposure (true,false default false): running exposure classification or not
  - --exposure_dataset (true, false, default false): use this boolean if you want to test origional kim et al. model with exposure blur images

```bash
python main.py --exposure true
```

## Running Real Time with ROS
ROSService for real time implementation in ros 

Sevice message called BlurOutput.srv
Input
- sensor_msgs/Image bgr : bgr image input 
- string save_name : file name including file extension for blur output image
Output
- bool success : service success of failure boolean
- std_msgs/Int32 label : argmax label for most type of blur used
- string msg : error message if service failed

launch file uses blur_detection_ros_main.py parameters include
- model_checkpoint: location of the weights .ckpt you want to use 
- height: input image height
- width: input image width

Following service message is called

```bash
rospy.ServiceProxy('/success_apc_blur_detection/blurdetection', BlurOutput)
```
 Follow the ROSService instructions for full description on how to send and receive messages

## License ##
TODO correct licence 

