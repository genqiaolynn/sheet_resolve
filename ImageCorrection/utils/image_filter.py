# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :image_filter
# @Date     :2020/12/9 0009
# @Author   :xieyi
-------------------------------------------------
"""
import cv2


class ImageFiltering:

    def __init__(self):
        pass

    @staticmethod
    def filtering(img_matrix, method="guassian"):
        """
        滤波处理，可以不处理
        :param img_matrix:
        :param method:
        :return:
        """
        if method == "mean":
            # 均值滤波
            img_filter = cv2.blur(img_matrix, (5, 5))
        elif method == "guassian":
            # 高斯滤波
            img_filter = cv2.GaussianBlur(img_matrix, (5, 5), 0)
        elif method == "median":
            # 中值滤波
            img_filter = cv2.medianBlur(img_matrix, 5)
        elif method == "bilater":
            # 双边滤波
            img_filter = cv2.bilateralFilter(img_matrix, 9, 75, 75)
        else:
            # 不处理直接返回
            img_filter = img_matrix
        return img_filter

    @staticmethod
    def image_gray(img_filter):
        """
        灰度图
        :param img_filter:
        :return:
        """
        if not len(img_filter.shape) == 2:
            img_gray = cv2.cvtColor(img_filter, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_filter
        # img_gray[img_gray > 250] = 250
        # img_gray[img_gray < 150] = 150
        return img_gray

    @staticmethod
    def image_binary_inv(img_gray):
        """
        图片二值化颜色取反
        :param img_gray:
        :return:
        """
        ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return img_binary

    @staticmethod
    def image_binary(img_gray):
        """
        图片二值化
        :param img_gray:
        :return:
        """
        ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_binary
