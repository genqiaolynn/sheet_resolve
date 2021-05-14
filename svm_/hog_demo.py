# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/1 14:33'

from skimage.feature import hog
from skimage import io
from PIL import Image
import cv2
import numpy as np

img = cv2.cvtColor(cv2.imread(r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-01-29\3.jpg'), cv2.COLOR_BGR2GRAY)
# normalised_blocks, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
#                                    block_norm='L2-Hys', visualize=True)
# io.imshow(hog_image)
# io.show()


# img = cv2.imread('F:/picture/desk/1.jpg')

v1 = cv2.Canny(img, 80, 150)  # minValue取值较小，边界点检测较多,maxValue越大，标准越高，检测点越少
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
cv2.imshow('res', res)
cv2.waitKey(20000)