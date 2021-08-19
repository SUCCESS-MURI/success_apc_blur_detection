import cv2
import numpy as np

image = cv2.imread('/home/mary/code/local_success_dataset/new_muri_data/updated_cropped_dataset/img_10.png')
salinecy = cv2.imread('/home/mary/code/U-2-Net/test_data/u2net_results/img_10.png')
salinecy = salinecy[:,:,0]
#cv2.equalizeHist(salinecy, salinecy)
# return opposite of color value
(T, saliency) = cv2.threshold(salinecy, 1, 1, cv2.THRESH_BINARY)
newimage = saliency[:,:,np.newaxis]*image

Corrected = 255*np.power((newimage*1.)/255,2.2)
# moton starts at 7 kernal size (this is for images with some level of sensitivity that doesnt have to be seen)
size = 7
angle = 0
k = np.zeros((size, size), dtype=np.float32)
k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
k = k * ( 1.0 / np.sum(k) )
blur = cv2.filter2D(Corrected, -1, k)
#blur = cv2.blur(Corrected,(17,17))
motion_blurred_image = np.power((blur*1.)/255,(1/2.2))

# focus starts at 7 kernal size (not this is because we dont have to have the highest level of detection for our focus blur)
blur = cv2.blur(Corrected,(7,7))
focus_blurred_image = np.power((blur*1.)/255,(1/2.2))

# range from 1.8 to 2.5
brightness_blur = cv2.convertScaleAbs(Corrected, alpha=1.8, beta=0)
# idxs = np.argwhere(image >= 245)[:, 0:2]
# for i in range(len(idxs)):
#     brightness_blur[idxs[i, 0], idxs[i, 1]] = [255, 255, 245]
brightness_blurred_image = np.power((brightness_blur*1.)/255,(1/2.2))
# range from 0.05 to 0.5
darkness_blur = cv2.convertScaleAbs(Corrected, alpha=0.5, beta=0)
darkness_blurred_image = np.power((darkness_blur*1.)/255,(1/2.2))


cv2.imshow('origional image',image)
# https://stackoverflow.com/questions/48331211/how-to-use-cv2-imshow-correctly-for-the-float-image-returned-by-cv2-distancet
cv2.imshow('gamma corrected blur motion', motion_blurred_image)
cv2.imshow('gamma corrected blur focus', focus_blurred_image)
cv2.imshow('gamma corrected blur brightness', brightness_blurred_image)
cv2.imshow('gamma corrected blur darkness', darkness_blurred_image)
cv2.waitKey(0)