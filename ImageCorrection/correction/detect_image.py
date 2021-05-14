# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :verify_image
# @Date     :2020/12/18 0018
# @Author   :xieyi
-------------------------------------------------
"""

from ImageCorrection.utils.projection import ProjectionSegmentImage


class DetectImageStatus:
    """检测图片状态：是否需要进行霍夫直线探测"""

    def __init__(self, img_sub_matrix, rotate_degrees=0.01):
        self.img_matrix = img_sub_matrix
        self.rotate_degrees = rotate_degrees
        self.projection_image = ProjectionSegmentImage()

    def check_image(self):
        """
        仿射变换的旋转：
        以中心点逆时针方向旋转(左旋)为正        anti_clock_wise
        以中心点顺时针方向旋转(右旋)为负        clock_wise
        :return:
        """
        degrees = 0
        image_matrix = self.img_matrix
        nonzero_projection = self.projection_image.get_rotate_projection(image_matrix, degrees)
        # show_console("验证图片", "角度{0} -- 投影值{1}".format(degrees, nonzero_projection))
        # 探测当前点左右角度值投影情况
        left_degrees = degrees + self.rotate_degrees
        left_nonzero_projection = self.projection_image.get_rotate_projection(image_matrix, left_degrees)
        # show_console("验证图片", "左旋 角度{0} -- 投影值{1}".format(left_degrees, left_nonzero_projection))
        right_degrees = degrees - self.rotate_degrees
        right_nonzero_projection = self.projection_image.get_rotate_projection(image_matrix, right_degrees)
        # show_console("验证图片", "右旋 角度{0} -- 投影值{1}".format(right_degrees, right_nonzero_projection))
        if nonzero_projection < min(left_nonzero_projection, right_nonzero_projection):
            image_status = True
            rotate_direction = 0
        else:
            image_status = False
            if left_nonzero_projection < right_nonzero_projection:
                rotate_direction = 1
            else:
                rotate_direction = -1
        return image_status, rotate_direction
