# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/4/29 15:06'


import cv2
import numpy as np
img_path = r'E:\111_4_26_test_img\images\1\201907141524_0001.jpg'
image = cv2.imread(img_path)
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
sobel = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, ksize=3)
ret, binary = cv2.threshold(sobel, 220, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

# 设置膨胀和腐蚀操作的核函数
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

# 膨胀一次，让轮廓突出
dilation = cv2.dilate(binary, element2, iterations=1)

# 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
erosion = cv2.erode(dilation, element1, iterations=1)

# aim = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,element1, 1 )   #此函数可实现闭运算和开运算
# 以上膨胀+腐蚀称为闭运算，具有填充白色区域细小黑色空洞、连接近邻物体的作用

# 再次膨胀，让轮廓明显一些
dilation2 = cv2.dilate(erosion, element2, iterations=3)

cv2.imwrite(r'E:\111_4_26_test_img\save\bb.jpg', dilation2)