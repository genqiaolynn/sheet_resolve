# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/3 16:04'


from brain_api_charge import get_ocr_text_and_coordinate11
import cv2, re
import numpy as np
from shapely.geometry import LineString, Polygon


'''
条码补全本身不难，就是根据ocr出来的条码几个字向四周扩散，直到找到黑色像素为止，但是需要过滤掉attention里面的条码
attention里面的条码可能索引不是从0开始的，所以要判断索引的起始点和终点
'''


def infer_bar_code_demo(img, words_result, attention_region):
    attention_polygon_list = []
    for attention in attention_region:
        coordinates = attention['bounding_box']
        xmin = coordinates['xmin']
        ymin = coordinates['ymin']
        xmax = coordinates['xmax']
        ymax = coordinates['ymax']
        attention_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])    # 四个点
        attention_polygon_list.append(attention_polygon)
    img_cols, img_rows = img.shape[0], img.shape[1]
    pattern = '条形码|条码|形码'
    for index, words in enumerate(words_result):
        words_string = words['words']
        chars_list = words['chars']
        key_string = [(ele.span(), ele.group()) for ele in re.finditer(pattern, words_string) if ele]
        if key_string:
            for key in key_string:
                start_index = key[0][0]
                end_index = key[0][1] - 1

    print(img_cols)


if __name__ == '__main__':
    img_path = r'E:\save\1\2.jpg'
    img = cv2.imread(img_path)
    # 原图和经过预处理的图像相比，经过预处理的图像ocr得到的结果丢失的信息更少一些
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_gray, 0, 235, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # ret, binary = cv2.threshold(img_gray, 0, 230, cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    dst = cv2.dilate(binary, kernel)  # 膨胀
    # dst = cv2.erode(dst, np.ones((1, 1), np.uint8))
    cv2.imwrite('dilate.jpg', dst)
    resp = get_ocr_text_and_coordinate11(dst)
    words_result = resp['words_result']
    attention_region = [{'class_name': 'attention', 'bounding_box': {'xmin': 809, 'ymin': 372, 'xmax': 1490, 'ymax': 715}, 'score': '0.9999'}]
    infer_bar_code_demo(img, words_result, attention_region)
