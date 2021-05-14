# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :correction
# @Date     :2020/12/18 0018
# @Author   :xieyi
-------------------------------------------------
"""
import cv2
from ImageCorrection.correction.detect_image import DetectImageStatus
from ImageCorrection.correction.image_collection import ImageCollection
from ImageCorrection.correction.py_hough_trans import HoughImage


def rotate_image(img_matrix, degrees=0):
    """
    根据degrees对图片matrix进行中心旋转
    :param img_matrix: 三维矩阵
    :param degrees: 角度
    :return:
    """
    if degrees != 0:
        (h, w) = img_matrix.shape[:2]
        (cx, cy) = (w // 2, h // 2)
        # 计算二维旋转的仿射变换矩阵
        m = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)
        img_dst_matrix = cv2.warpAffine(img_matrix, m, (w, h),
                                        # flags=cv2.INTER_NEAREST,
                                        # borderMode=cv2.BORDER_REPLICATE,
                                        borderValue=(255, 255, 255))
    else:
        img_dst_matrix = img_matrix
    # show_image("dst", img_dst_matrix)
    return img_dst_matrix


def correction_entrance(image):
    try:
        hough_image = HoughImage()
        image_collection = ImageCollection(image)
        img_original_matrix, img_canny_matrix, img_sub_projection_matrix, img_padding_projection_matrix = \
            image_collection.image_processing()
        detect_image = DetectImageStatus(img_padding_projection_matrix)
        image_status, rotate_direction = detect_image.check_image()
        if not image_status:
            optimal_degrees = hough_image.get_optimal_degrees(img_canny_matrix, img_sub_projection_matrix)
        else:
            optimal_degrees = 0
        # show_console("投影结果", "角度 ---------------------- {0}".format(optimal_degrees))
        return rotate_image(img_original_matrix, optimal_degrees)
    except Exception as ex:
        print('ex:', ex)
        return image
