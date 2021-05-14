# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :py_hough_trans
# @Date     :2020/12/11 0011
# @Author   :xieyi
-------------------------------------------------
"""
import numpy as np
import math, cv2
from ImageCorrection.utils.projection import ProjectionSegmentImage
from ImageCorrection.utils.show_utils import show_console, show_hough_lines


class HoughImage(object):
    """
    适合于对单张图片设置最低探测直线条数少的情况
    """

    def __init__(self, deviation_degrees=10):
        """
        :param deviation_degrees: 图片最大可能的旋转角度
        """
        # 最低探测直线条数
        self.critical_line_num = 10
        # 最小长宽的最大比率
        self.line_max_ratio = 0.2
        # 退出探测的最小比率
        self.line_min_ratio = 0.05
        # 为了探测更多条直线，每次下调比率
        self.decline_ratio = 0.01
        # 中文字符的高度
        self.zh_cn_height = 32
        self.deviation_degrees = deviation_degrees
        self.projection_image = ProjectionSegmentImage()

    def _search_optimal_degrees(self, img_sub_projection_matrix, degrees_list):
        """
        根据角度获取投影结果，并根据结果排序获取最优角度值
        :param img_sub_projection_matrix:
        :param degrees_list:
        :return:
        """
        non_repeat_degrees_list = sorted(list(set(degrees_list)))
        # show_console("无重复角度", "角度列表：{0}".format(non_repeat_degrees_list))
        w_h_degrees_list = []
        for non_repeat_degrees in non_repeat_degrees_list:
            nonzero_sub_projection = self.projection_image.get_rotate_projection(img_sub_projection_matrix,
                                                                                 non_repeat_degrees)
            w_h_degrees_list.append([nonzero_sub_projection, non_repeat_degrees])
            show_console("无重复角度", "角度{0} -- 投影值{1}".format(non_repeat_degrees, nonzero_sub_projection))
        wh_degrees_array = np.array(w_h_degrees_list)
        degrees_array = wh_degrees_array[np.argsort(wh_degrees_array[:, 0])]
        optimal_degrees = degrees_array[0][1]
        return optimal_degrees

    def _add_small_degrees(self, degrees_list, image_w, image_h):
        """
        增加由于误差问题带来的小角度
        :param degrees_list: 检测得到的角度集合
        :param image_w:
        :param image_h:
        :return:
        """
        degrees_list = sorted(list(set(degrees_list)))
        show_console("直线探测角度", "角度列表：{0}".format(degrees_list))
        max_degrees_loss = math.degrees(math.atan(self.zh_cn_height/image_w))
        show_console("新增小角度", "字符最小高度{0}--图片size{1}--角度{2}".format(
            self.zh_cn_height, (image_w, image_h), max_degrees_loss))
        new_degrees_list = []
        for degrees in degrees_list:
            if degrees > 0:
                for loss in np.arange(0, max_degrees_loss, 0.1):
                    new_degrees_list.append(float('% .2f' % (degrees - loss)))
            elif degrees < 0:
                for loss in np.arange(0, max_degrees_loss, 0.1):
                    new_degrees_list.append(float('% .2f' % (degrees + loss)))
            else:
                new_degrees_list.append(0)
        return new_degrees_list

    def _line_to_degrees(self, lines):
        """
        获取角度偏差在正负deviation_degrees度之间的角度
        :param lines: 探测获得的直线
        :return:
        """
        degrees_list = []
        for i in range(lines.shape[0]):
            line_theta = lines[i][0][1]
            degrees = line_theta / np.pi * 180 - 90
            if -1 * self.deviation_degrees < degrees < self.deviation_degrees:
                degrees_list.append(float('% .2f' % degrees))
        return degrees_list

    def _search_hough_lines(self, img_canny, image_w, image_h):
        # 获得至少10条直线,如果直线条数不足，每次下降0.01，直至比率小于0.01
        line_num = 0
        line_ratio = self.line_max_ratio
        lines = None
        while line_num < self.critical_line_num and line_ratio > self.line_min_ratio:
            lines = cv2.HoughLines(img_canny, 1, np.pi / 180, int(min(int(image_h), int(image_w)) * line_ratio))
            if lines is None:
                line_num = 0
            else:
                line_num = len(lines)
            show_console("直线探测", "当前比率: {0} -- 探测直线数量: {1}".format(line_ratio, line_num))
            line_ratio = line_ratio - self.decline_ratio
        return line_num, lines, line_ratio

    def get_optimal_degrees(self, img_canny_matrix, img_sub_projection_matrix):
        """
        根据霍夫直线探测更多直线，得到角度
        增加小角度，匹配最优角度返回
        :param img_canny_matrix: 原图的边缘检测图
        :param img_sub_projection_matrix:取图片的中心局部区域外边沿填充白色
        # :param img_padding_projection_matrix:取图片外边沿填充白色
        :return:
        """
        image_h, image_w = img_canny_matrix.shape
        line_num, lines, line_ratio = self._search_hough_lines(img_canny_matrix, image_w, image_h)
        show_hough_lines(img_canny_matrix, lines)
        if line_num > 0:
            degrees_list = self._add_small_degrees(self._line_to_degrees(lines), image_w, image_h)
            optimal_degrees = self._search_optimal_degrees(img_sub_projection_matrix, degrees_list)
        else:
            optimal_degrees = 0
        return optimal_degrees
