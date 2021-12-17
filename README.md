# Blur Detection and Classifcation that includes Brightness and Darkness Blur

This repository contains the training and test code for running our blur detection model based on the Kim et al. model with the additional 2 output labels for brightness and darkness blur definition. 
Links to our dataset include motion, defocus, brightness and darkness blurs together in each scene.

--------------------------
## Prerequisites (tested)
- Ubuntu 20.04
- Tensorflow-gpu 2.2.0
- Tensorlayer 2.2.3
- OpenCV2
- Listed in the document Packages_Requirements
- Using Ananconda Virtual Enviornment

### Note with Tensorflow update
read the tensorflow_issues_and_solutions document for code solutions for errors that might arise with updating tensorflow

## Kim et al. Train and Testing Details
- Go to XXXX to run the original Kim et al. Training and Testing code using CHUK and syntheic 

## Our Train Details 
- We updated the [CUHK blur detection dataset](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html) by adding brightness and darkness blur to the dataset
- We updated the Training and testing lists in the config.py.
- Run the following line in the environment:
  
```bash
python main.py --is_train true
```

## Test Details
- download [model weights](https://drive.google.com/file/d/11FBVmAIfeHDHpOjLXewzpA2lgcOOqo2_/view?usp=sharing) from google drive and save the model into 'model' folder.
- specify a path of input folder in 'main.py' at line #39
- run 'main.py --is_train false'

## Evaluation Details
- test.py does all of the miou, f1-score, accuracy and confusion matrices calculations. refer to the test.py script with details on the evaluation implementation. 
- Just run test.py to run evaluation

## Dataset 

## Creation
- Refer to XXXX and XXX for how the orgional CHUK and Kim et al. datasets were defined. 
- dataset/dataset_creation.py shows how we created our dataset with brightness and darkness. 

## CHUK Dataset 
- This link () - origional CHUK dataset 

## Kim et al. Dataset
- Refer to XXXX for details and download links for Kim et al. Datsets

## Brightness and Darkness Dataset
- This link () our training and testing images used for blur detection with brightness and darkness images 

## License ##
NEED to Update for MURI/CMU

This software is being made available under the terms in the [LICENSE](LICENSE) file.
Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

