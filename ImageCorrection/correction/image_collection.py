# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :image_collection
# @Date     :2020/12/21 0021
# @Author   :xieyi
-------------------------------------------------
"""
import copy
import cv2
import numpy as np
from ImageCorrection.utils.projection import ProjectionSegmentImage
from ImageCorrection.utils.image_filter import ImageFiltering
from ImageCorrection.utils.show_utils import show_image


class ImageCollection:
    """
    产生待使用的图片集合
    """

    def __init__(self, image):
        # img_original_matrix = cv2.imread(image_path)
        img_original_matrix = image
        show_image("original", img_original_matrix)
        self.img_original_matrix = img_original_matrix
        self.image_h = img_original_matrix.shape[0]
        self.image_w = img_original_matrix.shape[1]
        self.img_duplicate_matrix = copy.deepcopy(img_original_matrix)
        self.image_filtering = ImageFiltering()
        self.projection_image = ProjectionSegmentImage()

    def image_processing(self):
        # 获取图片array
        img_original_matrix = self.img_original_matrix
        img_duplicate_matrix = self.img_duplicate_matrix
        # 图片处理部分
        img_new_matrix = self.image_filtering.filtering(img_duplicate_matrix, method="guassian")
        show_image("filtering", img_new_matrix)

        img_gray_matrix = self.image_filtering.image_gray(img_new_matrix)
        show_image("gray", img_gray_matrix)

        img_binary_matrix = self.image_filtering.image_binary(img_gray_matrix)
        show_image("binary", img_binary_matrix)

        isw, iew, ish, ieh = self.projection_image.division_wh(img_binary_matrix)

        # 设定Canny参数范围
        # kmeans = KMeans(n_clusters=2)
        # kmeans.fit(img_gray_matrix.reshape(-1, 1))
        # print(kmeans.labels_)
        # print(kmeans.cluster_centers_)

        img_canny_matrix = cv2.Canny(img_gray_matrix[ish:ieh, isw:iew], 150, 250)
        show_image("canny", img_canny_matrix)

        # 若有必要可以使用子图
        img_sub_matrix = img_original_matrix[ish:ieh, isw:iew]
        show_image("sub", img_sub_matrix)

        # 投影图使用二值化图
        # 中心区域的切割图
        img_sub_projection_matrix = np.full((self.image_h, self.image_w), 255, dtype=np.uint8)
        img_sub_projection_matrix[ish:ieh, isw:iew] = img_binary_matrix[ish:ieh, isw:iew]
        show_image("sub_projection", img_sub_projection_matrix)
        # 周边填充投影图
        fh_idx = int(self.image_h * 1.2)
        fw_idx = int(self.image_w * 1.2)
        img_padding_projection_matrix = np.full((fh_idx, fw_idx), 255, dtype=np.uint8)
        h_idx = int(self.image_h * 0.1)
        w_idx = int(self.image_w * 0.1)
        img_padding_projection_matrix[h_idx:h_idx+self.image_h, w_idx:w_idx+self.image_w] = img_binary_matrix
        show_image("padding_projection", img_padding_projection_matrix)

        return img_original_matrix, img_canny_matrix, img_sub_projection_matrix, img_padding_projection_matrix
