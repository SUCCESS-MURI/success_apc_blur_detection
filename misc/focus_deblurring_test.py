
import numpy as np
# import matplotlib.pyplot as plt
from PIL import ImageEnhance
from PIL import Image
from matplotlib import pyplot as plt
# from scipy.misc import imfilter, imread
from scipy import ndimage
from skimage import color, data, restoration ,io
from scipy.signal import convolve2d as conv2
import cv2
import random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorlayer as tl
from skimage.morphology import disk
from scipy.ndimage import convolve


# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
from tensorflow.python.training import py_checkpoint_reader

from model import VGG19_pretrained, Decoder_Network_classification_3_labels
VGG_MEAN = [103.939, 116.779, 123.68]

gamma = 1.5

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
    psf = gaussian_kernal(  )#
    # Restore Image using wiener algorithm
    r += 0.1 * r.std() * np.random.standard_normal(r.shape)
    g += 0.1 * g.std() * np.random.standard_normal(g.shape)
    b += 0.1 * b.std() * np.random.standard_normal(b.shape)
    deconvolved_R ,_ = restoration.unsupervised_wiener(r, psf)
    deconvolved_G ,_ = restoration.unsupervised_wiener(g, psf)
    deconvolved_B ,_ = restoration.unsupervised_wiener(b, psf)
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
    return blurred ,kernel

def normalvariate_random_int(mean, variance, dmin, dmax):
    r = dmax + 1
    while r < dmin or r > dmax:
        r = int(random.normalvariate(mean, variance))
    return r

def uniform_random_int(dmin, dmax):
    r = random.randint(dmin ,dmax)
    return r

def random_motion_blur_kernel(mean=50, variance=15, dmin=10, dmax=100):
    random_degree = normalvariate_random_int(mean, variance, dmin, dmax)
    random_angle = uniform_random_int(-180, 180)
    return random_degree ,random_angle

# create out of focus blur for 3 channel images
def create_out_of_focus_blur(image ,kernelsize):
    kernal = disk(kernelsize)
    kernal = kernal /kernal.sum()
    image_blurred = np.stack([convolve(c ,kernal) for c in image.T]).T
    # image_blurred = pyblur.DefocusBlur(image,kernelsize)#cv2.blur(image,(kernelsize,kernelsize))
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
def get_weights(sess,network):
    # https://github.com/TreB1eN/InsightFace_Pytorch/issues/137
    dict_weights_trained = np.load('../final_model.npy',allow_pickle=True)[()]
    params = []
    keys = dict_weights_trained.keys()
    for weights in network.trainable_weights:
        name = weights.name
        splitName ='/'.join(name.split('/')[2:-1])
        count_layers = 0
        for key in keys:
            keySplit = '/'.join(key.split(',')[1:])
            if '/d' in keySplit:
                keySplit = '/'.join(keySplit.split('/')[:-2])
            if splitName == keySplit:
                if 'bias' in name.split('/')[-1]:
                    params.append(dict_weights_trained[key][count_layers+1][0])
                else:
                    params.append(dict_weights_trained[key][count_layers][0])
                break
            count_layers = count_layers + 2

    sess.run(tl.files.assign_weights(params, network))

if __name__ == '__main__':
    # lets make a brightness darkness, and motion and focus blur image
    image_input = io.imread('/home/mary/Documents/Reaserch/ICRA/image_48.png')#io.imread('/home/mary/Documents/Reaserch/ICRA/block_perception_0.5_1_rgb_image-32_outoffocus_cropped.png')#i
    # io.imsave('input_image_no_blur.png', image_input)
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # images = [image_input]
    # for i in range(1,4):
    #     image = cv2.filter2D(images[i-1], -1, kernel)
    #     images.append(image)
    # image_input_1 = io.imread('/home/mary/code/Images_ICRA/input_image_brightness_blur_input_muri_sharpned1.png')
    # image_input_2 = io.imread('/home/mary/code/Images_ICRA/input_image_brightness_blur_input_muri_sharpned2.png')
    # image_input_3 = io.imread('/home/mary/code/Images_ICRA/input_image_brightness_blur_input_muri_sharpned3.png')
    # images = [image_input ,image_input_1 ,image_input_2 ,image_input_3]

    # image_input = 255.0 * np.power((image_input * 1.) / 255.0, gamma)
    #
    # # make a motion blur image
    # #kernal_size,angle = random_motion_blur_kernel()
    # image_motion,motion_kernal = apply_motion_blur(image_input,7,150)
    #
    # kernelsize = 5
    # image_focus = create_out_of_focus_blur(image_input, kernelsize)
    #
    # image_brightness = create_brightness_and_darkness_blur(image_input, 2.3, 0)
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
    # io.imsave('input_image_brightness_blur_input_muri.png', images[0])

    ### DEFINE MODEL ###
    patches_blurred = tf.compat.v1.placeholder('float32', [1, image_input.shape[0], image_input.shape[1], 3],
                                               name='input_patches')
    labels = tf.compat.v1.placeholder('int64', [1, image_input.shape[0], image_input.shape[1], 1], name='labels')
    with tf.compat.v1.variable_scope('Unified'):
        with tf.compat.v1.variable_scope('VGG') as scope1:
            input, n, f0, f0_1, f1_2, f2_3 = VGG19_pretrained(patches_blurred, reuse=False, scope=scope1)
        with tf.compat.v1.variable_scope('UNet') as scope2:
            net_regression, _, _, _ = Decoder_Network_classification_3_labels(input, n, f0, f0_1, f1_2,
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
    image = np.copy(image_input)
    kernel = np.array([[-1,-1,-1], [-1,5,-1], [-1,-1,-1]])

    percentage_sharp_images = []

    for i in range(4):
        # now run through blur detection and classification
        if image.shape[2] > 3:
            image = image[: ,: ,0:3]
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
        label = np.argmax([num_0 ,num_1 ,num_2 ,num_3 ,num_4])
        num = num_1+num_3+num_2+num_4
        dem = num_1+num_3+num_2+num_4+num_0
        percentage_sharp_images.append((num/dem)*100)

        # gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
        # image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
        #image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        image = ImageEnhance.Sharpness(Image.fromarray(image)).enhance(1.5)
        image = np.array(image)
        io.imsave('/home/mary/code/local_success_dataset/IROS_Dataset/examples_for_autoexposure/ex5_outoffocus_sharpened_'+str(i+1)+'.png',image)
    percentage_sharp_images = np.array(percentage_sharp_images)
    plt.rc('font', size=30)  # controls default text size
    plt.rc('axes', titlesize=20)  # fontsize of the title
    plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=30)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=30)  # fontsize of the legend
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.plot(percentage_sharp_images,linewidth=6)
    plt.xlabel('# Times Image was Sharpened',fontweight='bold')
    plt.ylabel('% Blur Pixels',fontweight='bold')
    plt.yticks(np.arange(40, 70, 5))
    plt.xticks(np.arange(0, 4, 1))
    plt.grid(b=True)
    plt.show()
