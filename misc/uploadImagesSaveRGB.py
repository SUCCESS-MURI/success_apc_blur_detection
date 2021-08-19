import argparse
import cv2
import numpy as np
import tensorlayer as tl

# this takes the bag file and saves the rgb image
def upload_and_save_rgb(args):
    #image_input_list = []
    image_output_list = []
    #image_softmax_output = []
    #image_input_name_list = sorted(tl.files.load_file_list(path=args.location + '/input/', regx='/*.(png|PNG)',
    #                                                        printable=False))
    image_output_name_list = sorted(tl.files.load_file_list(path=args.location + '/output/', regx='/*.(png|PNG)',
                                                           printable=False))
    # image_softmax_name_list = sorted(tl.files.load_file_list(path=args.location + '/softmax/', regx='/*.(npy|NPY)',
    #                                                         printable=False))
    # for imageFileName in image_input_name_list:
    #     image_input_list.append(cv2.imread(args.location + '/input/'+imageFileName))
    for imageFileName in image_output_name_list:
        image_output_list.append(cv2.imread(args.location + '/output/'+imageFileName))
    # for imageFileName in image_softmax_name_list:
    #     image_softmax_output.append(np.load(args.location + '/softmax/'+imageFileName))
    for i in range(len(image_output_list)):
        image = image_output_list[i][:,:,0]
        imageName = image_output_name_list[i].split('/')[-1]
        # now color code
        rgb_blur_map = np.zeros((image.shape[0],image.shape[1],3))
        # blue motion blur
        rgb_blur_map[image == 1] = [255, 0, 0]
        # green focus blur
        rgb_blur_map[image == 2] = [0, 255, 0]
        # red darkness blur
        rgb_blur_map[image == 3] = [0, 0, 255]
        # pink brightness blur
        rgb_blur_map[image == 4] = [255, 192, 203]
        # yellow unknown blur
        rgb_blur_map[image == 5] = [0, 255, 255]
        cv2.imwrite(args.location + '/rgb/' + imageName, rgb_blur_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--location', type=str,
                        default='/home/mary/code/local_success_dataset/muri/muri_images_run/BYU_image_data')
    args = parser.parse_args()
    upload_and_save_rgb(args)
