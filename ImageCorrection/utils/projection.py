# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :projection
# @Date     :2020/12/9 0009
# @Author   :xieyi
------------------------------------------  -------
"""
import numpy as np
import cv2
from ImageCorrection.utils.image_filter import ImageFiltering


class ProjectionSegmentImage:

    def __init__(self):
        self.image_filtering = ImageFiltering()

    def _binary_image(self, image_matrix):
        img_gray = self.image_filtering.image_gray(image_matrix)
        image_binary = self.image_filtering.image_binary(img_gray)
        return image_binary

    @staticmethod
    def projection_matrix(image_binary, axis=0):
        """
        axis:0 == 垂直投影到W轴上
        axis:1 == 水平投影到H轴上
        :param image_binary:
        :param axis:
        :return:
        """
        image_h, image_w = image_binary.shape
        count_list = []
        if axis == 0:
            nonzero_min = image_h
            for w in range(image_w):
                count = image_h - np.count_nonzero(image_binary[:, w])
                count_list.append(count)
                if (count != 0) and (count < nonzero_min):
                    nonzero_min = count
        else:
            nonzero_min = image_w
            for h in range(image_h):
                count = image_w - np.count_nonzero(image_binary[h, :])
                count_list.append(count)
                if (count != 0) and (count < nonzero_min):
                    nonzero_min = count

        return count_list, nonzero_min

    @staticmethod
    def segment_list(count_list, nonzero_min):
        """
        遍历各个被分割的字符位置
        :param count_list:
        :param nonzero_min:
        :return:
        """
        begin, end = 0, 0
        interval_list = []
        idx = 1
        while idx < len(count_list):
            last_val = count_list[idx - 1]
            now_val = count_list[idx]
            if last_val < nonzero_min <= now_val:
                begin = idx
            if now_val < nonzero_min <= last_val:
                end = idx
                char_distance = end - begin
                interval_list.append([begin, end, char_distance])
            idx += 1
        return interval_list

    @staticmethod
    def _division_nonzero(count_list, nonzero_min, credible):
        """
        从前向后遍历第一个下降沿
        :param count_list:
        :param nonzero_min:
        :return:
        """
        division_id = 0
        idx = 1
        min_id = int(credible * len(count_list))
        if count_list[division_id] == 0:
            return division_id
        while idx < len(count_list):
            last_val = count_list[idx - 1]
            now_val = count_list[idx]
            if now_val < nonzero_min <= last_val:
                division_id = idx
                break
            idx += 1
        else:
            division_id = min_id
        if division_id > min_id:
            division_id = min_id
        return division_id

    @staticmethod
    def _division_last_nonzero(count_list, nonzero_min, credible):
        """
        从后向前遍历第一个上升沿
        :param count_list:
        :param nonzero_min:
        :return:
        """
        division_id = len(count_list)-1
        idx = len(count_list)-2
        max_idx = int(len(count_list) * (1 - credible))
        if count_list[division_id] == 0:
            return division_id
        while idx >= 0:
            now_val = count_list[idx]
            next_val = count_list[idx + 1]
            if now_val < nonzero_min <= next_val:
                division_id = idx
                break
            idx -= 1
        else:
            # 未找到
            division_id = max_idx
        # 找到却不适合
        if division_id < max_idx:
            division_id = max_idx
        return division_id

    def division_wh(self, image_binary, credible=0.1):
        w_count_list, nonzero_min = self.projection_matrix(image_binary, axis=0)
        isw = self._division_nonzero(w_count_list, nonzero_min, credible)
        iew = self._division_last_nonzero(w_count_list, nonzero_min, credible)
        h_count_list, nonzero_min = self.projection_matrix(image_binary, axis=1)
        ish = self._division_nonzero(h_count_list, nonzero_min, credible)
        ieh = self._division_last_nonzero(h_count_list, nonzero_min, credible)
        return isw, iew, ish, ieh

    def get_nonzero_w(self, image_binary):
        w_count_list, nonzero_min = self.projection_matrix(image_binary, axis=0)
        # print(w_count_list)
        nonzero_w = len([item for item in w_count_list if item > 0])
        return nonzero_w

    def get_nonzero_h(self, image_binary):
        h_count_list, nonzero_min = self.projection_matrix(image_binary, axis=1)
        # print(h_count_list)
        nonzero_h = len([item for item in h_count_list if item > 0])
        return nonzero_h

    def get_rotate_projection(self, img_matrix, degrees=0, method="wh"):
        (h, w) = img_matrix.shape[:2]
        if degrees != 0:
            (cx, cy) = (w // 2, h // 2)
            # 计算二维旋转的仿射变换矩阵
            m = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)
            img_dst_matrix = cv2.warpAffine(img_matrix, m, (w, h),
                                            # flags=cv2.INTER_NEAREST,
                                            # borderMode=cv2.BORDER_REPLICATE,
                                            borderValue=(255, 255, 255)
                                            )
        else:
            img_dst_matrix = img_matrix
        if len(img_dst_matrix.shape) > 2:
            image_binary = self._binary_image(img_dst_matrix)
        else:
            image_binary = img_dst_matrix
        nonzero_w = self.get_nonzero_w(image_binary)
        nonzero_h = self.get_nonzero_h(image_binary)
        if method == "w":
            return nonzero_w
        elif method == "iwh":
            return nonzero_w/w + nonzero_h/h
        else:
            return nonzero_w + nonzero_h


