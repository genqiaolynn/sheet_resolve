# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @File     :test_warp
# @Date     :2020/12/21 0021
# @Author   :xieyi
-------------------------------------------------
"""
import math

font_h = 32
image_w = 3340
radians = math.atan(font_h/image_w)
print("radians", radians)
degrees = math.degrees(radians)
print("degrees", degrees)

