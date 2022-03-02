
import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    paper = cv2.imread('/home/mary/code/NN_Blur_Detection_apc/output_iros_run_3/m3_resultsout_of_focus0288_0.png')#cv2.imread('/home/mary/code/local_success_dataset/IROS_Dataset/02_14_2022/Testing/images/out_of_focus0288_0.png')
    # Coordinates that you want to Perspective Transform
    # pts1 = np.float32([[0, 0], [480, 0], [0, 640], [480, 640]])
    # # Size of the Transformed Image
    # # t
    # pts2 = np.float32([[0, 15], [480, 5], [0, 620], [480, 635]])

    pts1 = np.float32([[0, 0], [60, 0], [0, 80], [60, 80]])
    # Size of the Transformed Image
    # t
    pts2 = np.float32([[0, 2], [60, .4], [0, 78], [60, 79.9]])

    # pts1 = np.float32([[0, 0], [240, 0], [0, 320], [240, 320]])
    # # Size of the Transformed Image
    # # t
    # pts2 = np.float32([[0, 8], [240, 2], [0, 310], [240, 318]])

    # n
    #pts2 = np.float32([[0, 2], [490, 10], [0, 640], [490, 622]])
    #pts2 = np.float32([[0, 0], [480, 20], [0, 640], [480, 620]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(paper, M,(80,60),borderValue =[255,255,255])#(640,480)
    plt.imshow(dst)
    cv2.imwrite('output_m3_new_wraped_image.png',dst)