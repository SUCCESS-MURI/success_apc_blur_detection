import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rc('font', size=20) #controls default text size
plt.rc('axes', titlesize=20) #fontsize of the title
plt.rc('axes', labelsize=20) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
plt.rc('legend', fontsize=30) #fontsize of the legend

plt.clf()
# # kim et all all of the images
# final_confmatrix = np.array([[0.397,0.046,0.557],[0.145,0.77,0.085],[0.092,0.023,0.885]])
# np.save('all_labels_results_conf_matrix_kimetal.npy', final_confmatrix)
# cax=plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
# classNames = ['No Blur', 'Motion', 'Focus']
# #plt.title('Our Weights - Test Data Confusion Matrix')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# tick_marks = np.arange(len(classNames))
# plt.xticks(tick_marks, classNames, rotation=45)
# plt.yticks(tick_marks, classNames)
# # for i in range(len(classNames)):
# #     for j in range(len(classNames)):
# #         plt.text(j, i, str(final_confmatrix[i][j]))
# thresh = final_confmatrix.max() / 2.
# for i in range(final_confmatrix.shape[0]):
#     for j in range(final_confmatrix.shape[1]):
#         plt.text(j, i, format(final_confmatrix[i, j]),
#                  ha="center", va="center",
#                  color="white" if final_confmatrix[i, j] > thresh else "black")
# plt.tight_layout()
# cbar = plt.colorbar(cax,ticks=[final_confmatrix.min(),final_confmatrix.max()])
# cbar.ax.set_yticklabels(['0.0', ' 1.0'])
# plt.savefig('conf_matrix_all_labels_kimetal.png')
# plt.show()
#
# plt.clf()
# # kim et all all labels brightness and darkness images only
# final_confmatrix = np.array([[0.236,0.033,0.731],[0.0,0.0,0.0],[0.0,0.0,0.0]])
# np.save('all_labels_results_conf_matrix_kimetal_bd.npy', final_confmatrix)
# final_confmatrix = np.round(final_confmatrix, 3)
# plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
# classNames = ['No Blur', 'Motion', 'Focus']
# #plt.title('Our Weights - Test Data Confusion Matrix')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# tick_marks = np.arange(len(classNames))
# plt.xticks(tick_marks, classNames, rotation=45)
# plt.yticks(tick_marks, classNames)
# # for i in range(len(classNames)):
# #     for j in range(len(classNames)):
# #         plt.text(j, i, str(final_confmatrix[i][j]))
# thresh = final_confmatrix.max() / 2.
# for i in range(final_confmatrix.shape[0]):
#     for j in range(final_confmatrix.shape[1]):
#         plt.text(j, i, format(final_confmatrix[i, j]),
#                  ha="center", va="center",
#                  color="white" if final_confmatrix[i, j] > thresh else "black")
# plt.tight_layout()
# cbar = plt.colorbar(ticks=[final_confmatrix.min(),final_confmatrix.max()])
# cbar.ax.set_yticklabels(['0.0', ' 1.0'])
# plt.savefig('conf_matrix_all_labels_kimetal_bd.png')
# plt.show()

plt.clf()
# our all labels all images
final_confmatrix = np.round(np.load('/home/mary/code/NN_Blur_Detection_apc/output_cuhk_updated_Run_2_resized_.5/all_labels_results_conf_matrix.npy'),3)
#np.save('all_labels_results_conf_matrix_ourweights.npy', final_confmatrix)
plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
classNames = ['No Blur', 'Motion', 'Focus','Darkness','Brightness']
#plt.title('Our Weights - Test Data Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
# for i in range(len(classNames)):
#     for j in range(len(classNames)):
#         plt.text(j, i, str(final_confmatrix[i][j]))
thresh = final_confmatrix.max() / 2.
for i in range(final_confmatrix.shape[0]):
    for j in range(final_confmatrix.shape[1]):
        plt.text(j, i, format(final_confmatrix[i, j]),
                 ha="center", va="center",
                 color="white" if final_confmatrix[i, j] > thresh else "black")
plt.tight_layout()
cbar = plt.colorbar(ticks=[final_confmatrix.min(),final_confmatrix.max()])
cbar.ax.set_yticklabels(['0.0', ' 1.0'])
plt.savefig('conf_matrix_all_labels_ourweights.png')
plt.show()

# plt.clf()
# # our all labels bd images
# final_confmatrix = np.array([[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],
#                              [0.0,0.0,0.0,0.0,0.0],[0.001,0.0,0.0,0.999,0.0],
#                              [0.193,0.002,0.026,0.001,0.779]])
# np.save('all_labels_results_conf_matrix_ourweights_bd.npy', final_confmatrix)
# plt.imshow(final_confmatrix, interpolation='nearest', cmap=plt.cm.Blues)
# classNames = ['No Blur', 'Motion', 'Focus','Darkness','Brightness']
# #plt.title('Our Weights - Test Data Confusion Matrix')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# tick_marks = np.arange(len(classNames))
# plt.xticks(tick_marks, classNames, rotation=45)
# plt.yticks(tick_marks, classNames)
# # for i in range(len(classNames)):
# #     for j in range(len(classNames)):
# #         plt.text(j, i, str(final_confmatrix[i][j]))
# thresh = final_confmatrix.max() / 2.
# for i in range(final_confmatrix.shape[0]):
#     for j in range(final_confmatrix.shape[1]):
#         plt.text(j, i, format(final_confmatrix[i, j]),
#                  ha="center", va="center",
#                  color="white" if final_confmatrix[i, j] > thresh else "black")
# plt.tight_layout()
# cbar = plt.colorbar(ticks=[final_confmatrix.min(),final_confmatrix.max()])
# cbar.ax.set_yticklabels(['0.0', ' 1.0'])
# plt.savefig('conf_matrix_all_labels_ourweights_bd.png')
# plt.show()
# count = 1