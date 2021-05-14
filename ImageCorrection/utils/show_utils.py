# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :show_utils
# @Date     :2020/12/18 0018
# @Author   :xieyi
-------------------------------------------------
"""
import cv2
import numpy as np
__not_release__ = False


def show_console(title, content):
    if __not_release__:
        print("{0} -- {1}".format(title, content))


def show_image(win_name, cv_image, multiple_size=0.25):
    if __not_release__:
        cv2.namedWindow(win_name, 0)
        win_h = int(cv_image.shape[0] * multiple_size)
        win_w = int(cv_image.shape[1] * multiple_size)
        cv2.resizeWindow(win_name, win_w, win_h)
        cv2.imshow(win_name, cv_image)
        cv2.waitKey(0)


def show_hough_lines(img_matrix, lines):
    if lines is None:
        return
    if __not_release__:
        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                cv2.line(img_matrix, (x1, y1), (x2, y2), (0, 0, 255), 2)
        show_image("hough_lines", img_matrix)
