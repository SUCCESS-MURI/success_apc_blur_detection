# Exposure Deep Blur Detection and Classification Dataset Creation and Misc Scripts

This repository contains the code for creating the datasets used for exposure training/testing. Please do not release this code.  

--------------------------

## Dataset Creation

### Split Dataset Scripts
 Both split_dataset.py and split_dataset_blur.py are variations of taking a list of images and spliting them randomly into training and testing 
Note this is only used for gathered datasets this was not used for CUHK & PHOS - those were already split by kim et al. specifics for CUHK and random splits for PHOS.
arguments are as follows
- --file_input_path: input path where images are located - Note: gt and RGB images are located in their names folders: gt & image accordingly
- --file_output_path: output path where images will be put - Note: gt and RGB images will be placed in their names folders: gt & images accordingly
For split_dataset_blur.py, this splits the datasets that had labeled - normal/no blur, motion, focus, over and underexposure

Run the scripts in the following format example
```bash
python split_dataset.py --file_input_path /your/input/images/path --file_output_path /your/output/path
```

### IROS/Exposure Real & Synthetic 

dataset_creation.py has all the code that is used for creating real, synthetic training and testing
Please read the code to really understand the specifics but in general training is doing:
- Get all images
- Degamma them and get their saliency mask images - Will have to run saliency masking (U2Net) before running this code
- all steps simultaneously getting gt mask of each blur
- get a random base focus image 
- overlay with motion blur object
- overlay with either real underexposure blur or synthetic underexposure blur
- overlay with either real overexposure blur or synthetic overexposure blur
- gamma back image and save 
- same for mask/gt 
- repeat process above for x amount of times
Testing is just labeling full images with appropriate gt labels

Line 26 is gamma value - find this specific for dataset - CHUK and PHOS used 2.2 - standard

apply_motion_blur - synthetic motion blur maker

line 62 and 67 have default values for random over-and underexposure (brightness - overexposure, darkness - underexposure) synthetic creation - feel free to change these ranges

#### Training functions 

create_dataset_for_training - used for creating real exposure overlay
create_dataset_for_training_synthetic - used for creating synthetic exposure overlay 
create_dataset_for_training_synthetic_and_real - used for creating synthetic and real exposure overlay - note this is set up for allowing real exposure obejcts to be different from syntheic exposure objects - used for training variation 4 

#### Testing functions
create_dataset_for_testing - not used
create_dataset_for_testing_2 - used for creating general testing images
create_dataset_for_testing_my_real_bd_images_test4 - used for exposure testing datasets
misc scripts for testing can be used if found useful

Inputs are not all used in each variation of training or testing. Feel free to change and modify these for personal use

### IPS Real & Synthetic

Need the utils.py script from [repo](https://github.com/SUCCESS-MURI/success_apc_blur_detection/releases/tag/v2.0.0) to run these scripts

#### Training functions

dataset_creation_training_fetch.py has all the code that is used for creating real, synthetic training code
Please read the code to really understand the specifics but in general it is doing what is mentioned above but with our ips recorded blur dataset. 

Line 25 is gamma value - specific for this dataset - 1.5

line 43 and 48 have default values for random over-and underexposure (brightness - overexposure, darkness - underexposure) synthetic creation - feel free to change these ranges

create_dataset_for_training_type_1_motion - used for creating real exposure overlay
create_dataset_for_training_type_2_motion - used for creating synthetic exposure overlay 
create_dataset_for_training_type_3_motion - used for creating synthetic and real exposure overlay 

Other creation datasets were tries but fails and are archived for possible future use

Inputs that are used
- --data_dir: input folder location of images/gt - Note look at all labels for images (focus, saliency etc) need to be present for code to run
- --output_data_dir: output folder location. will have two folders images, and gt accordingly
- --data_extension: for image extension changes default .png
- --training_type: 1, 2, or 3 for each type that is described above 

#### Testing scripts

dataset_creation_testing_fetch.py has all the code that is used for creating the testing dataset
Please read the code to really understand the specifics but in general it is doing what is mentioned above but with our ips recorded blur dataset.

create_dataset_for_testing_motion - used for creating general testing images
other testing functions can be used if found useful

Inputs that are used
- --data_dir: input folder location of images/gt - Note look at all labels for images (focus, saliency etc) need to be present for code to run
- --output_data_dir: output folder location. will have two folders images, and gt accordingly
- --data_extension: for image extension changes default .png

## MISC

bd_ranges.py - created ranges for syntheic dataset creation

focus_deblurring_test.py - performs experiment for seeing how exposure correction is not the same as focus correction

perspectiveTransform.py - used for warping images for use in network diagram

plotting.py - when plotting failed allowed for not haveing to run all testing to recreate images

recordBagImages.py - used for recording images from bag files for use in blur testing

test_gamma.py - used for testing gamma values and correction

uploadImagesDaveRGB.py - for use in getting blur output from labels 0 to 5 to rgb colors

