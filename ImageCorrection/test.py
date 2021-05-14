# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :test
# @Date     :2020/12/10 0010
# @Author   :xieyi
-------------------------------------------------
"""
from ImageCorrection.correction.correction import correction_entrance
import time
import cv2
"""单张图片的测试"""


if __name__ == "__main__":
    st = time.time()
    image_path = "static/jpg/10000009.jpg"
    image_matrix = correction_entrance(image_path)
    print(image_matrix)
    et = time.time()
    print("耗时时间{0}秒".format(float('%.4f' % (et - st))))
