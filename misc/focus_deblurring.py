import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#from scipy.misc import imfilter, imread
from skimage import color, data, restoration,io
from scipy.signal import convolve2d as conv2
import cv2
import random
import tensorflow as tf
import tensorlayer as tl
from skimage.morphology import disk
from scipy.ndimage import convolve


# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
from tensorflow.python.training import py_checkpoint_reader

from model import VGG19_pretrained, Decoder_Network_classification, Origional_Decoder_Network_classification
from setup.loadNPYWeightsSaveCkpt import get_weights_checkpoint, get_weights

VGG_MEAN = [103.939, 116.779, 123.68]

gamma = 2.2

def gaussian_kernal(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

# decompose rgb
def decompose_rgba(image):
    red_image = image[:, :, 0]
    green_image = image[:, :, 1]
    blue_image = image[:, :, 2]
    return red_image, green_image, blue_image

def correct_blur_through_deconvolution_rl(image):
    b, g, r = decompose_rgba(image)
    r = (r - np.min(r)) * 1 / (np.max(r) - np.min(r))
    g = (g - np.min(g)) * 1 / (np.max(g) - np.min(g))
    b = (b - np.min(b)) * 1 / (np.max(b) - np.min(b))
    psf = gaussian_kernal()
    # Restore Image using Richardson-Lucy algorithm
    deconvolved_R = restoration.richardson_lucy(r, psf, iterations=50)
    deconvolved_G = restoration.richardson_lucy(g, psf, iterations=50)
    deconvolved_B = restoration.richardson_lucy(b, psf, iterations=50)
    corrected_image = np.empty(image.shape)
    corrected_image[:, :, 0] = deconvolved_R
    corrected_image[:, :, 1] = deconvolved_G
    corrected_image[:, :, 2] = deconvolved_B
    return corrected_image

def correct_blur_through_deconvolution_wiener(image):
    r, g, b = decompose_rgba(image)
    r = (r - np.min(r)) * 1 / (np.max(r) - np.min(r))
    g = (g - np.min(g)) * 1 / (np.max(g) - np.min(g))
    b = (b - np.min(b)) * 1 / (np.max(b) - np.min(b))
    psf = gaussian_kernal()#
    # Restore Image using wiener algorithm
    r += 0.1 * r.std() * np.random.standard_normal(r.shape)
    g += 0.1 * g.std() * np.random.standard_normal(g.shape)
    b += 0.1 * b.std() * np.random.standard_normal(b.shape)
    deconvolved_R,_ = restoration.unsupervised_wiener(r, psf)
    deconvolved_G,_ = restoration.unsupervised_wiener(g, psf)
    deconvolved_B,_ = restoration.unsupervised_wiener(b, psf)
    corrected_image = np.empty(image.shape)
    corrected_image[:, :, 0] = deconvolved_R
    corrected_image[:, :, 1] = deconvolved_G
    corrected_image[:, :, 2] = deconvolved_B
    return corrected_image

# https://github.com/Imalne/Defocus-and-Motion-Blur-Detection-with-Deep-Contextual-Features/blob/a368a3e0a8869011ec167bb1f8eb82ceff091e0c/DataCreator/Blend.py#L14
def apply_motion_blur(image, size, angle):
    Motion = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = np.diag(np.ones(size))
    kernel = cv2.warpAffine(kernel, Motion, (size, size))
    kernel = kernel / size
    blurred = cv2.filter2D(image, -1, kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return blurred,kernel

def normalvariate_random_int(mean, variance, dmin, dmax):
    r = dmax + 1
    while r < dmin or r > dmax:
        r = int(random.normalvariate(mean, variance))
    return r

def uniform_random_int(dmin, dmax):
    r = random.randint(dmin,dmax)
    return r

def random_motion_blur_kernel(mean=50, variance=15, dmin=10, dmax=100):
    random_degree = normalvariate_random_int(mean, variance, dmin, dmax)
    random_angle = uniform_random_int(-180, 180)
    return random_degree,random_angle

# create out of focus blur for 3 channel images
def create_out_of_focus_blur(image,kernelsize):
    kernal = disk(kernelsize)
    kernal = kernal/kernal.sum()
    image_blurred = np.stack([convolve(c,kernal) for c in image.T]).T
    #image_blurred = pyblur.DefocusBlur(image,kernelsize)#cv2.blur(image,(kernelsize,kernelsize))
    return image_blurred

def create_brightness_and_darkness_blur(image, alpha, beta):
    new_img = image * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img

# def correct_blur_denoise_nl_mean(image):
#     image_gamma = 255.0 * np.power((image * 1.) / 255.0, gamma)
#     sigma_est = np.mean(restoration.estimate_sigma(image_gamma, multichannel=True))
#     patch_kw = dict(patch_size=5,      # 5x5 patches
#                 patch_distance=6,  # 13x13 search area
#                 multichannel=True)
#     # fast algorithm
#     denoiseImage = restoration.denoise_nl_means(image_gamma, h=0.6 * sigma_est, sigma=sigma_est,
#                                      fast_mode=True, **patch_kw,preserve_range=True)
#     cv2.normalize(denoiseImage, denoiseImage, 0, 255, cv2.NORM_MINMAX)
#     denoiseImage = np.round(np.power((denoiseImage * 1.) / 255.0, (1.0 / gamma)) * 255.0).astype(np.uint8)
#     denoiseImage = denoiseImage.astype(np.uint8)
#     return denoiseImage
#
# def correct_blur_through_denoise_means_then_deconv(image,kernal):
#     denoised_image = correct_blur_denoise_nl_mean(image)
#     corrected_image = correct_blur_through_deconvolution_wiener(denoised_image,kernal)
#     cv2.normalize(corrected_image, corrected_image, 0, 255, cv2.NORM_MINMAX)
#     corrected_image = corrected_image.astype(np.uint8)
#     return corrected_image

if __name__ == '__main__':
    # lets make a brightness darkness, and motion and focus blur image
    image_input = io.imread('/home/mary/code/Images_ICRA/input_image_brightness_blur_input_muri.png')#io.imread('/home/mary/code/local_success_dataset/CHUK_Dataset/08_25_2021/Testing/images/motion0007_0.png')
    # io.imsave('input_image_no_blur.png', image_input)
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # images = [image_input]
    # for i in range(1,4):
    #     image = cv2.filter2D(images[i-1], -1, kernel)
    #     images.append(image)
    image_input_1 = io.imread('/home/mary/code/Images_ICRA/input_image_brightness_blur_input_muri_sharpned1.png')
    image_input_2 = io.imread('/home/mary/code/Images_ICRA/input_image_brightness_blur_input_muri_sharpned2.png')
    image_input_3 = io.imread('/home/mary/code/Images_ICRA/input_image_brightness_blur_input_muri_sharpned3.png')
    images = [image_input,image_input_1,image_input_2,image_input_3]

    #image_input = 255.0 * np.power((image_input * 1.) / 255.0, gamma)
    #
    # # make a motion blur image
    # #kernal_size,angle = random_motion_blur_kernel()
    # image_motion,motion_kernal = apply_motion_blur(image_input,7,150)
    #
    # kernelsize = 5
    # image_focus = create_out_of_focus_blur(image_input, kernelsize)
    #
    #image_brightness = create_brightness_and_darkness_blur(image_input, 2.3, 0)
    # image_darkness = create_brightness_and_darkness_blur(image_input, 0.1, -10)

    # image_motion = (image_motion - np.min(image_motion)) * (1.0 - (0.0)) / \
    #                (np.max(image_motion) - np.min(image_motion)) + 0.0
    # image_focus = (image_focus - np.min(image_focus)) * (1.0 - (0.0)) / \
    #                (np.max(image_focus) - np.min(image_focus)) + 0.0
    # image_brightness = (image_brightness - np.min(image_brightness)) * (1.0 - (0.0)) / \
    #               (np.max(image_brightness) - np.min(image_brightness)) + 0.0
    # image_darkness = (image_darkness - np.min(image_darkness)) * (1.0 - (0.0)) / \
    #                    (np.max(image_darkness) - np.min(image_darkness)) + 0.0
    #
    # images = [np.round(np.power((image_motion * 1.), (1.0 / gamma)) * 255.0),
    #           np.round(np.power((image_focus * 1.), (1.0 / gamma)) * 255.0),
    #           np.round(np.power((image_darkness * 1.), (1.0 / gamma)) * 255.0),
    #           np.round(np.power((image_brightness * 1.), (1.0 / gamma)) * 255.0)]

    # io.imsave('input_image_motion_blur_input.png', images[0])
    # io.imsave('input_image_focus_blur_input.png', images[1])
    # io.imsave('input_image_darkness_blur_input.png', images[2])
    #io.imsave('input_image_brightness_blur_input_muri.png', images[0])

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [1, image_input.shape[0], image_input.shape[1], 3],
                                               name='input_patches')
    labels = tf.compat.v1.placeholder('int64', [1, image_input.shape[0], image_input.shape[1], 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, _, _, _ = Origional_Decoder_Network_classification(input, n, f0, f0_1, f1_2,
                                                                                            f2_3, reuse=False,
                                                                                            scope=scope2)

    output_map = tf.expand_dims(tf.math.argmax(tf.nn.softmax(net_regression.outputs), axis=3), axis=3)
    # Load checkpoint
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    # Load checkpoint
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    configTf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    configTf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=configTf)
    tl.layers.initialize_global_variables(sess)
    # read check point weights
    # reader = py_checkpoint_reader.NewCheckpointReader('../model/SA_net_cuhk_updated_Run_2.ckpt')
    # state_dict = {v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()}
    # # save weights to the model
    # get_weights_checkpoint(sess, net_regression, state_dict)
    # Load checkpoint
    get_weights(sess, net_regression)
    print("loaded all the weights")

    for image in images:
        # now run through blur detection and classifcation
        if image.shape[2] > 3:
            image = image[:,:,0:3]
        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]
        bgr = np.zeros(image.shape)
        bgr[:, :, 0] = blue - VGG_MEAN[0]
        bgr[:, :, 1] = green - VGG_MEAN[1]
        bgr[:, :, 2] = red - VGG_MEAN[2]

        blur_map = sess.run([output_map], {net_regression.inputs: np.expand_dims((bgr), axis=0)})[0]

        blurmap_flap = blur_map.flatten()
        num_0 = np.sum(blurmap_flap == 0)
        num_1 = np.sum(blurmap_flap == 1)
        num_2 = np.sum(blurmap_flap == 2)
        num_3 = np.sum(blurmap_flap == 3)
        num_4 = np.sum(blurmap_flap == 4)
        label = np.argmax([num_0,num_1,num_2,num_3,num_4])

        # if label == 0:
        #     print("image is not blurred")
        # if label == 1:
        #     print("image is motion blurred. Will be corrected")
        #     #image_gamma = 255.0 * np.power((image * 1.) / 255.0, gamma)
        #     #image_gamma = 255.0 * np.power((image_gamma * 1.) / 255.0, gamma)
        #     corrected_image = correct_blur_through_deconvolution_wiener(np.copy(np.round(image).astype(np.uint8)))
        #     corrected_image = (corrected_image - np.min(corrected_image)) * (255.0 - (0.0)) / (np.max(corrected_image) - np.min(corrected_image)) + 0.0
        #     #cv2.normalize(corrected_image, corrected_image, 0, 255, cv2.NORM_MINMAX)
        #     corrected_image = corrected_image.astype(np.uint8)
        #     red = corrected_image[:, :, 0]
        #     green = corrected_image[:, :, 1]
        #     blue = corrected_image[:, :, 2]
        #     bgr = np.zeros(image.shape)
        #     bgr[:, :, 0] = blue - VGG_MEAN[0]
        #     bgr[:, :, 1] = green - VGG_MEAN[1]
        #     bgr[:, :, 2] = red - VGG_MEAN[2]
        #
        #     blur_map = sess.run([output_map], {net_regression.inputs: np.expand_dims((bgr), axis=0)})[0]
        #
        #     blurmap_flap = blur_map.flatten()
        #     num_0 = np.sum(blurmap_flap == 0)
        #     num_1 = np.sum(blurmap_flap == 1)
        #     num_2 = np.sum(blurmap_flap == 2)
        #     num_3 = np.sum(blurmap_flap == 3)
        #     num_4 = np.sum(blurmap_flap == 4)
        #     new_label = np.argmax([num_0, num_1, num_2, num_3, num_4])
        #     print('now our new label is '+str(new_label))
        #     # display both images in a 1x2 grid
        #     # fig = plt.figure()
        #     # # create a new figure
        #     # fig.add_subplot(1, 2, 1)
        #     # # draw first image
        #     # plt.imshow(image/255)
        #     # #plt.gray()
        #     # fig.add_subplot(1, 2, 2)
        #     # # draw second image
        #     # plt.imshow(corrected_image)
        #     # # saves current figure as a PNG file
        #     # plt.show()
        #     io.imsave('motion_img_correction_wiener.png',corrected_image)
        #     # displays figure
        # if label == 2:
        #     # focus
        #     print("Image is Focus blurred. Will be corrected")
        #     # need a focus blur correction
        #     #corrected_image = correct_blur_through_deconvolution_wiener(np.copy(np.round(image).astype(np.uint8)))
        #     # corrected_image = cv2.GaussianBlur(image, (0,0), 3)
        #     # corrected_image = cv2.addWeighted(image, 1.5, corrected_image, -0.5, 0)
        #     #def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        #     # kernel_size = (5, 5)
        #     # sigma = 1.0
        #     # amount = 2.0
        #     # threshold = 0
        #     # """Return a sharpened version of the image, using an unsharp mask."""
        #     # blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        #     # sharpened = float(amount + 1) * image - float(amount) * blurred
        #     # sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        #     # sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        #     # sharpened = sharpened.round().astype(np.uint8)
        #     # if threshold > 0:
        #     #     low_contrast_mask = np.absolute(image - blurred) < threshold
        #     #     np.copyto(sharpened, image, where=low_contrast_mask)
        #     corrected_image = correct_blur_through_deconvolution_wiener(np.copy(np.round(image).astype(np.uint8)))
        #     corrected_image = (corrected_image - np.min(corrected_image)) * (255.0 - (0.0)) / (
        #                 np.max(corrected_image) - np.min(corrected_image)) + 0.0
        #     # cv2.normalize(corrected_image, corrected_image, 0, 255, cv2.NORM_MINMAX)
        #     corrected_image = corrected_image.astype(np.uint8)
        #     red = corrected_image[:, :, 0]
        #     green = corrected_image[:, :, 1]
        #     blue = corrected_image[:, :, 2]
        #     bgr = np.zeros(image.shape)
        #     bgr[:, :, 0] = blue - VGG_MEAN[0]
        #     bgr[:, :, 1] = green - VGG_MEAN[1]
        #     bgr[:, :, 2] = red - VGG_MEAN[2]
        #
        #     blur_map = sess.run([output_map], {net_regression.inputs: np.expand_dims((bgr), axis=0)})[0]
        #
        #     blurmap_flap = blur_map.flatten()
        #     num_0 = np.sum(blurmap_flap == 0)
        #     num_1 = np.sum(blurmap_flap == 1)
        #     num_2 = np.sum(blurmap_flap == 2)
        #     num_3 = np.sum(blurmap_flap == 3)
        #     num_4 = np.sum(blurmap_flap == 4)
        #     new_label = np.argmax([num_0, num_1, num_2, num_3, num_4])
        #     print('now our new label is ' + str(new_label))
        #     # cv2.imshow('deconvolution', res)
        #     # print deconvolved
        #     # display both images in a 1x2 grid
        #     # fig = plt.figure()
        #     # # create a new figure
        #     # fig.add_subplot(1, 2, 1)
        #     # # draw first image
        #     # plt.imshow(image/255)
        #     # plt.gray()
        #     # fig.add_subplot(1, 2, 2)
        #     # # draw second image
        #     # plt.imshow(corrected_image)
        #     # # saves current figure as a PNG file
        #     # plt.show()
        #     io.imsave('focus_img_correction_wiener.png', corrected_image)
        # if label == 3:
        #     # focus
        #     print("image is darkness blurred. Need to turn on light or use hdr")
        # if label == 4:
        #     # focus
        #     print("image is brightness blurred. Need to turn off light or use hdr")



