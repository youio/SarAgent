from sklearn.model_selection import train_test_split
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

greypath = "/home/qferguson/rockdata/grey/"
rockpath = "./rockmasks/"
sempath = "/home/qferguson/rockdata/semantic/"

grayfiles = os.listdir(greypath)
semfiles = os.listdir(sempath)

for file in os.listdir(sempath):
    img = cv2.imread(sempath+file, cv2.IMREAD_COLOR)
    maskimg = np.zeros(img.shape[:2], np.uint8)
    mask = np.all(img == [42, 59, 108], axis=2)
   # print(mask)
    maskimg[mask] = 255

    '''
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(maskimg)
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.show()'''

    #cv2.imwrite(rockpath+file, maskimg)
    # cv2.imshow("mask", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


greytrain, greyval, semtrain, semval = train_test_split(grayfiles, semfiles, test_size=0.2,random_state=42 )

for file in greytrain:
    shutil.copy(greypath+file, "./input/train_imgs/")

for file in greyval:
    shutil.copy(greypath+file, "./input/val_imgs/")

for file in semtrain:
    shutil.copy(rockpath+file, "./input/train_masks")

for file in semval:
     shutil.copy(rockpath+file, "./input/val_masks")
