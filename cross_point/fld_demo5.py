# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/4/24 19:33'


import cv2, time, heapq, os, copy
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon, LineString
import xml.etree.ElementTree as ET
from utils import *

# class_above_edge = ['solve', 'solve0', 'attention', 'choice', 'cloze', 'composition', 'composition0', 'correction']
class_above_edge = ['solve', 'solve0', 'attention', 'choice', 'composition', 'composition0', 'correction']  # 暂时不考虑cloze


def clean_repeat_lines(lines):
    lines_temp = lines.copy()
    lines_temp_arr = np.array(lines_temp)
    index1 = np.where(lines_temp_arr[0] < 0)[0]
    if index1:
        index1_list = index1.tolist()
        for i in index1_list:
            lines_temp_arr[i][0] = -lines_temp_arr[0]   # rho
            lines_temp_arr[i][1] = lines_temp_arr[1] - np.pi   # theta
    newlines = []
    newlines.append(lines.pop(5))
    for line in lines:
        flag = 0
        for newline in newlines:
            if((abs(line[0]-newline[0])<10) & (abs(line[1]-newline[1]) < 0.1)):
                flag = 1
        if(flag == 0):
            newlines.append(line)
    # print(newlines)
    return newlines


def get_direction(points_and_lines, box_polygon, image):
    image_height, image_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # raw
    cv2.imwrite(os.path.join(r'E:\December\math_12_18\1_18\test_img\save3\\', 'THRESH_BINARY_INV' + '.jpg'), binary)

    # DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = 1, 2, 4, 8
    DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = False, False, False, False

    # c++ 规则
    # CROSS_TL = DIR_UP | DIR_LEFT
    # CROSS_TR = DIR_UP | DIR_RIGHT
    # CROSS_TLR = DIR_UP | DIR_LEFT | DIR_RIGHT
    # CROSS_BL = DIR_DOWN | DIR_LEFT
    # CROSS_BR = DIR_DOWN | DIR_RIGHT
    # CROSS_BLR = DIR_DOWN | DIR_LEFT | DIR_RIGHT
    # CROSS_TBL = DIR_UP | DIR_DOWN | DIR_LEFT
    # CROSS_TBR = DIR_UP | DIR_DOWN | DIR_RIGHT
    # CROSS_TBLR = DIR_UP | DIR_DOWN | DIR_LEFT | DIR_RIGHT

    # python 规则
    # CROSS_TL = DIR_UP and DIR_LEFT
    # CROSS_TR = DIR_UP and DIR_RIGHT
    # CROSS_TLR = DIR_UP and DIR_LEFT and DIR_RIGHT
    # CROSS_BL = DIR_DOWN and DIR_LEFT
    # CROSS_BR = DIR_DOWN and DIR_RIGHT
    # CROSS_BLR = DIR_DOWN and DIR_LEFT and DIR_RIGHT
    # CROSS_TBL = DIR_UP and DIR_DOWN and DIR_LEFT
    # CROSS_TBR = DIR_UP and DIR_DOWN and DIR_RIGHT
    # CROSS_TBLR = DIR_UP and DIR_DOWN and DIR_LEFT and DIR_RIGHT
    # rule_list = [CROSS_TL, CROSS_TR, CROSS_TLR, CROSS_BL, CROSS_BR, CROSS_BLR, CROSS_TBL, CROSS_TBR, CROSS_TBLR]


    # 因为延长的线可能出问题,这里用的是原始的线
    print('points_and_lines:', points_and_lines)
    points = points_and_lines[0]
    raw_line = points_and_lines[1]

    point_x = int(points[0])
    point_y = int(points[1])
    # if point_x == 1620 and point_y == 3260:

    bbox = box_polygon.bounds
    left_boarder = int(bbox[0]) - point_x if int(bbox[0]) - point_x > 0 else 0
    right_boarder = int(bbox[2]) + point_x if int(bbox[2]) + point_x < image_width else image_width
    top_boarder = int(bbox[1]) - point_y if int(bbox[1]) - point_y > 0 else 0
    bottom_boarder = int(bbox[3]) + point_y if int(bbox[3]) + point_y < image_height else image_height

    cv2.rectangle(image, (left_boarder, top_boarder), (right_boarder, bottom_boarder), (255, 0, 255), 1)

    raw_line_lan_ = raw_line[0]
    raw_line_lon_ = raw_line[1]

    if abs(raw_line_lan_.bounds[2] - point_x) >= abs(raw_line_lan_.bounds[0] - point_x):
        raw_line_lan = LineString([(point_x, raw_line_lan_.bounds[1]), (raw_line_lan_.bounds[2], raw_line_lan_.bounds[3])])
    else:
        raw_line_lan = LineString([(raw_line_lan_.bounds[0], raw_line_lan_.bounds[1]), (point_x, raw_line_lan_.bounds[3])])

    if abs(raw_line_lon_.bounds[3] - point_y) <= abs(point_y - raw_line_lon_.bounds[1]):
        raw_line_lon = LineString([(raw_line_lon_.bounds[0], raw_line_lon_.bounds[1]), (raw_line_lon_.bounds[2], point_y)])
    else:
        raw_line_lon = LineString([(raw_line_lon_.bounds[0], point_y), (raw_line_lon_.bounds[2], raw_line_lon_.bounds[3])])

    # box_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    right_area = Polygon([(point_x, point_y - 10), (right_boarder, point_y - 10),
                          (right_boarder, point_y + 10), (point_x, point_y + 10)])

    # 横线
    cv2.line(image, (int(raw_line_lan.bounds[0]), int(raw_line_lan.bounds[1])),
             (int(raw_line_lan.bounds[2]), int(raw_line_lan.bounds[3])), (255, 0, 0), 1, cv2.LINE_AA)

    # cv2.line(image, (int(raw_line_lan_.bounds[0]), int(raw_line_lan_.bounds[1])),
    #          (int(raw_line_lan_.bounds[2]), int(raw_line_lan_.bounds[3])), (255, 255, 0), 1, cv2.LINE_AA)

    # 纵线
    cv2.line(image, (int(raw_line_lon.bounds[0]), int(raw_line_lon.bounds[1])),
             (int(raw_line_lon.bounds[2]), int(raw_line_lon.bounds[3])), (255, 0, 0), 1, cv2.LINE_AA)

    # cv2.line(image, (int(raw_line_lon_.bounds[0]), int(raw_line_lon_.bounds[1])),
    #          (int(raw_line_lon_.bounds[2]), int(raw_line_lon_.bounds[3])), (255, 255, 0), 1, cv2.LINE_AA)

    cv2.rectangle(image, (int(right_area.bounds[0]), int(right_area.bounds[1])),
                  (int(right_area.bounds[2]), int(right_area.bounds[3])), (255, 0, 255), 1)

    cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\extend_lines_by_bbox.jpg', image)

    # 水平向右
    decision = [raw_line_lan.within(right_area), raw_line_lan.contains(right_area), raw_line_lan.crosses(right_area)]
    if True in decision:
        DIR_RIGHT = True

    # 水平向左
    left_area = Polygon([(left_boarder, point_y - 10), (point_x, point_y - 10),
                         (point_x, point_y + 10), (left_boarder, point_y + 10)])
    decision = [raw_line_lan.within(left_area), raw_line_lan.contains(left_area), raw_line_lan.crosses(left_area)]
    if True in decision:
        DIR_LEFT = True

    # box_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    # 垂直向上
    top_area = Polygon([(point_x - 10, top_boarder), (point_x + 10, top_boarder),
                        (point_x + 10, point_y), (point_x - 10, point_y)])
    decision = [raw_line_lon.within(top_area), raw_line_lon.contains(top_area), raw_line_lon.crosses(top_area)]
    if True in decision:
        DIR_UP = True

    # 垂直向下
    bottom_area = Polygon([(point_x - 10, point_y), (point_x + 10, point_y),
                        (point_x + 10, bottom_boarder), (point_x - 10, bottom_boarder)])
    decision = [raw_line_lon.within(bottom_area), raw_line_lon.contains(bottom_area), raw_line_lon.crosses(bottom_area)]
    if True in decision:
        DIR_DOWN = True

    CROSS_TL = DIR_UP and DIR_LEFT
    CROSS_TR = DIR_UP and DIR_RIGHT
    CROSS_TLR = DIR_UP and DIR_LEFT and DIR_RIGHT
    CROSS_BL = DIR_DOWN and DIR_LEFT
    CROSS_BR = DIR_DOWN and DIR_RIGHT
    CROSS_BLR = DIR_DOWN and DIR_LEFT and DIR_RIGHT
    CROSS_TBL = DIR_UP and DIR_DOWN and DIR_LEFT
    CROSS_TBR = DIR_UP and DIR_DOWN and DIR_RIGHT
    CROSS_TBLR = DIR_UP and DIR_DOWN and DIR_LEFT and DIR_RIGHT

    direction_dict = {
        0: 'CROSS_TL',
        1: 'CROSS_TR',
        2: 'CROSS_TLR',
        3: 'CROSS_BL',
        4: 'CROSS_BR',
        5: 'CROSS_BLR',
        6: 'CROSS_TBL',
        7: 'CROSS_TBR',
        8: 'CROSS_TBLR',
    }
    rule_list_tmp = [CROSS_TL, CROSS_TR, CROSS_TLR, CROSS_BL, CROSS_BR, CROSS_BLR, CROSS_TBL, CROSS_TBR, CROSS_TBLR]
    index_ = np.where(np.array(rule_list_tmp) == True)[0]
    if len(index_) == 0:
        print('没找到方向:', points)
        return None
    else:
        print('index_:', index_)
        direction = direction_dict[index_[0]]
        return direction


def fld_demo1(image, sheet_dict, split_x):
    image_height, image_width = image.shape[:2]
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(img_gray)
    lines = [line[0] for line in dlines.tolist()]

    lines_arr = np.array(lines)
    width_ = lines_arr[:, 2] - lines_arr[:, 0]
    height_ = lines_arr[:, 3] - lines_arr[:, 1]
    lines_index1 = np.where(width_ > 50)[0]
    lines_index2 = np.where(height_ > 50)[0]
    lines_index = np.hstack([lines_index1, lines_index2])
    new_lines = [lines[ele] for ele in lines_index]
    new_lines = clean_repeat_lines(new_lines)
    split_x.insert(0, 0)
    split_x.append(image_width)

    # 对原始的raw_lines进行分栏
    line_block = []
    for index, split in enumerate(split_x[1:]):
        left_boarder, right_boarder = split_x[index], split_x[index + 1]
        top_boarder, bottom_boarder = 1, image_height - 1

        block_list = []
        for line in new_lines:
            # if round(line[2]) < round(line[0]) and round(line[3]) > round(line[1]):
            #     line_new = [line[0], line[1], line[0], line[3]]
            # elif round(line[2]) > round(line[0]) and round(line[3]) < round(line[1]):
            #     line_new = [line[0], line[1], line[2], line[1]]
            # else:
            #     line_new = line
            line_new = line
            line_tmp = LineString([(line_new[0], line_new[1]), (line_new[2], line_new[3])])
            polygon_tmp = Polygon([(left_boarder, top_boarder), (right_boarder, top_boarder),
                                   (right_boarder, bottom_boarder), (left_boarder, bottom_boarder)])
            decision = [polygon_tmp.contains(line_tmp),
                        polygon_tmp.within(line_tmp),
                        polygon_tmp.overlaps(line_tmp)]
            if True in decision:
                block_list.append(line_new)
        block_dict = {}
        block_dict['boarder'] = [left_boarder, top_boarder, right_boarder, bottom_boarder]
        block_dict['block_list'] = block_list
        line_block.append(block_dict)
    # print(line_block)

    # 对原始的raw_lines分横排还是纵排
    landscape_block_list = []
    longitudinal_block_list = []
    points_list = []
    for line in line_block:
        block_list = line['block_list']
        boarder = line['boarder']
        for index, ele in enumerate(block_list):
            if int(ele[2] - int(ele[0])) < 10:   # 纵排
                raw_line = LineString([(ele[0], ele[1]), (ele[2], ele[3])])
                extend_line =LineString([(ele[0], boarder[1]), (ele[0], boarder[3])])   # 纵排的时候y轴垂直
                points_list.extend([1, image_height - 1])
                line_start, line_end = boarder[1], boarder[3]
                # print('landscape')
            elif int(ele[3] - int(ele[1])) < 10:   # 横排
                raw_line = LineString([(ele[0], ele[1]), (ele[2], ele[3])])
                extend_line =LineString([(boarder[0], ele[1]), (boarder[2], ele[1])])   # 横排的时候x轴垂直
                points_list.extend([1, image_width - 1])
                line_start, line_end = boarder[0], boarder[2]
                # print('longitudinal')

            # cv2.line(image, (int(raw_line.coords[0][0]), int(raw_line.coords[0][1])),
            #          (int(raw_line.coords[1][0]), int(raw_line.coords[1][1])), (255, 255, 0), 2, cv2.LINE_AA)
            # cv2.line(image, (int(extend_line.coords[0][0]), int(extend_line.coords[0][1])),
            #          (int(extend_line.coords[1][0]), int(extend_line.coords[1][1])), (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\\' + str(index) + '_extend_lines' + '.jpg', image)
            # print('ok')

            # ele = LineString([(ele[0], ele[1]), (ele[2], ele[3])])
            # cv2.line(image, (int(ele.coords[0][0]), int(ele.coords[0][1])),
            #          (int(ele.coords[1][0]), int(ele.coords[0][1])), (0, 255, 0), 1, cv2.LINE_AA)

            # cv2.line(image, (int(line[0]), top_boarder), (int(line[2]), bottom_boarder), (255, 255, 0), 1, cv2.LINE_AA)

            # cv2.line(image, (int(ele[0]), int(ele[1])), (int(ele[2]), int(ele[3])), (255, 255, 0), 2, cv2.LINE_AA)
            # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\\' + str(index) + '_longitudinal' + '.jpg', image)
            # print(line)
    # for dline in new_lines:
    #     x0 = int(round(dline[0]))
    #     y0 = int(round(dline[1]))
    #     x1 = int(round(dline[2]))
    #     y1 = int(round(dline[3]))
    #     cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\raw_lines.jpg', image)


def filter_long_distance_lines_raw(lines, sheet_dict):
    # 去掉roi里面的线
    lines_tmp1 = []
    for index, region_box in enumerate(sheet_dict):
        coordinates = region_box['bounding_box']
        xmin = coordinates['xmin']
        ymin = coordinates['ymin']
        xmax = coordinates['xmax']
        ymax = coordinates['ymax']
        box_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        for line in lines:
            line_tmp = LineString([(line[0], line[1]), (line[2], line[3])])
            # line_tmp = LineString([(750, 502), (1035, 502)])
            decision = [line_tmp.within(box_polygon), line_tmp.contains(box_polygon), line_tmp.overlaps(box_polygon)]
            # attention:[179, 440, 1092, 729]
            # 横线 750, 502, 1035, 509
            if True in decision:
                lines_tmp1.append(line)
    lines_tmp = [line for line in lines if line not in lines_tmp1]
    return lines_tmp


def filter_long_distance_lines(lines, sheet_dict):
    # 去掉roi里面的线  也去掉mark里的线
    lines_tmp1 = []
    for index, region_box in enumerate(sheet_dict):
        if region_box['class_name'] in class_above_edge:
            coordinates = region_box['bounding_box']
            xmin = coordinates['xmin'] + 10
            ymin = coordinates['ymin'] + 10
            xmax = coordinates['xmax'] - 10
            ymax = coordinates['ymax'] - 10
            box_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            for line in lines:
                line_tmp = LineString([(line[0], line[1]), (line[2], line[3])])
                # line_tmp = LineString([(750, 502), (1035, 502)])
                # decision = [line_tmp.within(box_polygon), line_tmp.contains(box_polygon), line_tmp.overlaps(box_polygon)]
                decision = [line_tmp.within(box_polygon), line_tmp.contains(box_polygon)]
                # attention:[179, 440, 1092, 729]
                # 横线 750, 502, 1035, 509
                if True in decision:
                    lines_tmp1.append(line)
        elif region_box['class_name'] == 'mark':
            coordinates = region_box['bounding_box']
            xmin = coordinates['xmin']
            ymin = coordinates['ymin']
            xmax = coordinates['xmax']
            ymax = coordinates['ymax']
            box_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            for line in lines:
                line_tmp = LineString([(line[0], line[1]), (line[2], line[3])])
                # line_tmp = LineString([(750, 502), (1035, 502)])
                decision = [line_tmp.within(box_polygon), line_tmp.contains(box_polygon)]
                # attention:[179, 440, 1092, 729]
                # 横线 750, 502, 1035, 509
                if True in decision:
                    lines_tmp1.append(line)
    lines_tmp = [line for line in lines if line not in lines_tmp1]
    return lines_tmp


def fld_demo2(image, sheet_dict, split_x):
    image_height, image_width = image.shape[:2]
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(img_gray)
    lines = [line[0] for line in dlines.tolist()]

    lines_arr = np.array(lines)
    width_ = lines_arr[:, 2] - lines_arr[:, 0]
    height_ = lines_arr[:, 3] - lines_arr[:, 1]
    lines_index1 = np.where(width_ > 50)[0]
    lines_index2 = np.where(height_ > 50)[0]
    lines_index = np.hstack([lines_index1, lines_index2])
    new_lines = [lines[ele] for ele in lines_index]
    new_lines = clean_repeat_lines(new_lines)
    # new_lines = filter_long_distance_lines(new_lines, sheet_dict)    # TODO 未写完
    split_x.insert(0, 0)
    split_x.append(image_width)

    # 对原始的raw_lines进行分栏
    line_block = []
    for index, split in enumerate(split_x[1:]):
        left_boarder, right_boarder = split_x[index], split_x[index + 1]
        top_boarder, bottom_boarder = 1, image_height - 1

        block_list = []
        for line in new_lines:
            # if round(line[2]) < round(line[0]) and round(line[3]) > round(line[1]):
            #     line_new = [line[0], line[1], line[0], line[3]]
            # elif round(line[2]) > round(line[0]) and round(line[3]) < round(line[1]):
            #     line_new = [line[0], line[1], line[2], line[1]]
            # else:
            #     line_new = line
            line_new = line
            line_tmp = LineString([(line_new[0], line_new[1]), (line_new[2], line_new[3])])
            polygon_tmp = Polygon([(left_boarder, top_boarder), (right_boarder, top_boarder),
                                   (right_boarder, bottom_boarder), (left_boarder, bottom_boarder)])
            decision = [polygon_tmp.contains(line_tmp),
                        polygon_tmp.within(line_tmp),
                        polygon_tmp.overlaps(line_tmp)]
            if True in decision:
                block_list.append(line_new)
        block_dict = {}
        block_dict['boarder'] = [left_boarder, top_boarder, right_boarder, bottom_boarder]
        block_dict['block_list'] = block_list
        line_block.append(block_dict)
    # print(line_block)

    # 对原始的raw_lines分横排还是纵排
    landscape_block_list = []
    longitudinal_block_list = []
    points_list = []
    for l in range(len(line_block)):   # l代表栏数
        boarder = line_block[l]['boarder']
        block_list = line_block[l]['block_list']
        block_list1 = block_list.copy()
        block_list2 = block_list.copy()
        for i, ele1 in enumerate(block_list1):
            for j, ele1 in enumerate(block_list2):
                print('ok')



        for line in line_block:
            block_list = line['block_list']
            boarder = line['boarder']
            for index, ele in enumerate(block_list):
                if int(ele[2] - int(ele[0])) < 10:   # 纵排
                    raw_line = LineString([(ele[0], ele[1]), (ele[2], ele[3])])
                    extend_line =LineString([(ele[0], boarder[1]), (ele[0], boarder[3])])   # 纵排的时候y轴垂直
                    points_list.extend([1, image_height - 1])
                    line_start, line_end = boarder[1], boarder[3]
                    # print('landscape')
                elif int(ele[3] - int(ele[1])) < 10:   # 横排
                    raw_line = LineString([(ele[0], ele[1]), (ele[2], ele[3])])
                    extend_line =LineString([(boarder[0], ele[1]), (boarder[2], ele[1])])   # 横排的时候x轴垂直
                    points_list.extend([1, image_width - 1])
                    line_start, line_end = boarder[0], boarder[2]

    # for dline in new_lines:
    #     x0 = int(round(dline[0]))
    #     y0 = int(round(dline[1]))
    #     x1 = int(round(dline[2]))
    #     y1 = int(round(dline[3]))
    #     cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\raw_lines.jpg', image)


def filter_line_in_mark(lines, sheet_dict):
    # 去掉roi里面的线
    lines_tmp1 = []
    for index, region_box in enumerate(sheet_dict):
        class_name = region_box['class_name']
        if region_box['class_name'] == 'mark':
            coordinates = region_box['bounding_box']
            xmin = coordinates['xmin']
            ymin = coordinates['ymin']
            xmax = coordinates['xmax']
            ymax = coordinates['ymax']
            box_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            for line in lines:
                line_tmp = LineString([(line[0], line[1]), (line[2], line[3])])
                # line_tmp = LineString([(750, 502), (1035, 502)])
                decision = [line_tmp.within(box_polygon), line_tmp.contains(box_polygon), line_tmp.overlaps(box_polygon)]
                # attention:[179, 440, 1092, 729]
                # 横线 750, 502, 1035, 509
                if True in decision:
                    lines_tmp1.append(line)
    lines_tmp = [line for line in lines if line not in lines_tmp1]
    for line in lines_tmp:
        cv2.line(image, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\mark.jpg', image)
    return lines_tmp


# TODO 找距离ROI最近的线,这样做可以不分栏
def fld_demo3(image, sheet_dict, split_x):
    image_height, image_width = image.shape[:2]
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(img_gray)
    lines = [line[0] for line in dlines.tolist()]

    lines_arr = np.array(lines)
    width_ = lines_arr[:, 2] - lines_arr[:, 0]
    height_ = lines_arr[:, 3] - lines_arr[:, 1]
    lines_index1 = np.where(width_ > 50)[0]
    lines_index2 = np.where(height_ > 50)[0]
    lines_index = np.hstack([lines_index1, lines_index2])
    new_lines = [lines[ele] for ele in lines_index]
    new_lines = clean_repeat_lines(new_lines)
    lines_tmp = filter_long_distance_lines(new_lines, sheet_dict)
    # lines_without_mark = filter_line_in_mark(new_lines, sheet_dict)

    # 检验滤线这一步骤有没有问题
    # for dline in lines_tmp:
    #     x0 = int(round(dline[0]))
    #     y0 = int(round(dline[1]))
    #     x1 = int(round(dline[2]))
    #     y1 = int(round(dline[3]))
    #     cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    #     text = str([x0, y0, x1, y1])
    #     cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\raw_lines.jpg', image)

    # get roi polygon 适当放大一点区域,使的线能包含在面里
    roi_box_polygon = []
    for index, region_box in enumerate(sheet_dict):
        if region_box['class_name'] in class_above_edge:
            coordinates = region_box['bounding_box']
            xmin = coordinates['xmin'] - 5
            ymin = coordinates['ymin'] - 5
            xmax = coordinates['xmax'] + 5
            ymax = coordinates['ymax'] + 5
            box_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            roi_box_polygon.append(box_polygon)
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\enlarge_box.jpg', image)

    # print(roi_box_polygon)

    # 对line进行分横向纵向
    landscape_list = []       # 横向
    longitudinal_list = []    # 纵向
    for ele in lines_tmp:
        if int(ele[2]) - int(ele[0]) < 10:     # 纵向
            landscape_list.append(ele)
        elif int(ele[3]) - int(ele[1]) < 10:   # 横向
            longitudinal_list.append(ele)
    # print(landscape_list)

    # get lines nearest roi
    # 这里需要分横向纵向,可以不分栏解决问题
    nearest_line_from_roi = []
    for index, roi_object in enumerate(roi_box_polygon):
        landscape_distance = []
        longitudinal_diatance = []

        landscape_length = []
        longitudinal_length = []

        for line in landscape_list:
            line_tmp = LineString([(line[0], line[1]), (line[2], line[3])])
            distance1 = roi_object.distance(line_tmp)
            landscape_distance.append(distance1)
            landscape_length.append(line_tmp.length)

        for line in longitudinal_list:
            line_tmp = LineString([(line[0], line[1]), (line[2], line[3])])
            distance1 = roi_object.distance(line_tmp)
            longitudinal_diatance.append(distance1)
            longitudinal_length.append(line_tmp.length)

        # 排序规则是离ROI最近的index
        landscape_distance_min = np.argsort(landscape_distance)
        longitudinal_diatance_min = np.argsort(longitudinal_diatance)

        landscape1 = [landscape_list[ele] for ele in list(landscape_distance_min[:2])]
        longitudinal1 = [longitudinal_list[ele] for ele in list(longitudinal_diatance_min[:2])]

        # 排序规则是线最长的
        landscape_long_line = np.argsort(landscape_length)
        longitudinal_long_line = np.argsort(longitudinal_length)

        # 横排
        # lan_dis_and_long_len = landscape_distance_min + landscape_long_line
        # double_len = landscape_long_line * 2
        #
        # length1 = np.arange(0, len(lan_dis_and_long_len))
        # landscape_long_distance = length1[lan_dis_and_long_len == double_len]
        # landscape1 = [landscape_long_line[ele] for ele in list(landscape_long_distance)]
        # landscape11 = [landscape_list[ele] for ele in list(landscape1)]
        #
        # # 纵排
        # longitu_dis_and_long_len_ = longitudinal_diatance_min + longitudinal_long_line
        # double_len_ = longitudinal_long_line * 2
        #
        # length2 = np.arange(0, len(longitu_dis_and_long_len_))
        # longitu_long_distance_ = length2[longitu_dis_and_long_len_ == double_len_]
        # longitu1 = [longitudinal_diatance[ele] for ele in list(longitu_long_distance_)]
        # longitu11 = [longitudinal_list[ele] for ele in list(longitu1)]

        for ele1 in landscape1:
            cv2.line(image, (int(ele1[0]), int(ele1[1])), (int(ele1[2]), int(ele1[3])), (0, 255, 0), 2, cv2.LINE_AA)
        for ele2 in longitudinal1:
            cv2.line(image, (int(ele2[0]), int(ele2[1])), (int(ele2[2]), int(ele2[3])), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\\' + str(index) + '_extend_lines' + '.jpg', image)
        print('ok')
        # cv2.line(image, (int(landscape_min[0]), int(landscape_min[1])), (int(landscape_min[2]), int(landscape_min[3])), (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.line(image, (int(longitudinal_min[0]), int(longitudinal_min[1])), (int(longitudinal_min[2]), int(longitudinal_min[3])), (255, 255, 0), 2, cv2.LINE_AA)
        # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\\' + str(index) + '_extend_lines' + '.jpg', image)

        # # 堆排序,取最小的几个数
        # # min_dis = heapq.nsmallest(10, distance_list)
        # s_index = np.argsort(distance_list)
        # for index, ele in enumerate(list(s_index)):
        #     line = lines_tmp[ele]
        #     cv2.line(image, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 255, 0), 2, cv2.LINE_AA)
        #     cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\\' + str(index) + '_extend_lines' + '.jpg', image)
        #     print(s_index)

    # # 对原始的raw_lines进行分栏
    # line_block = []
    # for index, split in enumerate(split_x[1:]):
    #     left_boarder, right_boarder = split_x[index], split_x[index + 1]
    #     top_boarder, bottom_boarder = 1, image_height - 1
    #
    #     block_list = []
    #     for line in new_lines:
    #         # if round(line[2]) < round(line[0]) and round(line[3]) > round(line[1]):
    #         #     line_new = [line[0], line[1], line[0], line[3]]
    #         # elif round(line[2]) > round(line[0]) and round(line[3]) < round(line[1]):
    #         #     line_new = [line[0], line[1], line[2], line[1]]
    #         # else:
    #         #     line_new = line
    #         line_new = line
    #         line_tmp = LineString([(line_new[0], line_new[1]), (line_new[2], line_new[3])])
    #         polygon_tmp = Polygon([(left_boarder, top_boarder), (right_boarder, top_boarder),
    #                                (right_boarder, bottom_boarder), (left_boarder, bottom_boarder)])
    #         decision = [polygon_tmp.contains(line_tmp),
    #                     polygon_tmp.within(line_tmp),
    #                     polygon_tmp.overlaps(line_tmp)]
    #         if True in decision:
    #             block_list.append(line_new)
    #     block_dict = {}
    #     block_dict['boarder'] = [left_boarder, top_boarder, right_boarder, bottom_boarder]
    #     block_dict['block_list'] = block_list
    #     line_block.append(block_dict)
    # # print(line_block)
    #
    # # 对原始的raw_lines分横排还是纵排
    # landscape_block_list = []
    # longitudinal_block_list = []
    # points_list = []
    # for l in range(len(line_block)):   # l代表栏数
    #     boarder = line_block[l]['boarder']
    #     block_list = line_block[l]['block_list']
    #     block_list1 = block_list.copy()
    #     block_list2 = block_list.copy()
    #     for i, ele1 in enumerate(block_list1):
    #         for j, ele1 in enumerate(block_list2):
    #             print('ok')
    #
    #
    #
    #     for line in line_block:
    #         block_list = line['block_list']
    #         boarder = line['boarder']
    #         for index, ele in enumerate(block_list):
    #             if int(ele[2] - int(ele[0])) < 10:   # 纵排
    #                 raw_line = LineString([(ele[0], ele[1]), (ele[2], ele[3])])
    #                 extend_line =LineString([(ele[0], boarder[1]), (ele[0], boarder[3])])   # 纵排的时候y轴垂直
    #                 points_list.extend([1, image_height - 1])
    #                 line_start, line_end = boarder[1], boarder[3]
    #                 # print('landscape')
    #             elif int(ele[3] - int(ele[1])) < 10:   # 横排
    #                 raw_line = LineString([(ele[0], ele[1]), (ele[2], ele[3])])
    #                 extend_line =LineString([(boarder[0], ele[1]), (boarder[2], ele[1])])   # 横排的时候x轴垂直
    #                 points_list.extend([1, image_width - 1])
    #                 line_start, line_end = boarder[0], boarder[2]

    # for dline in lines_tmp:
    #     x0 = int(round(dline[0]))
    #     y0 = int(round(dline[1]))
    #     x1 = int(round(dline[2]))
    #     y1 = int(round(dline[3]))
    #     cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\raw_lines.jpg', image)


def fld_demo4(image, sheet_dict, split_x):
    img = Image.fromarray(image)
    t = time.time()
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(img_gray)
    lines = [line[0] for line in dlines.tolist()]

    lines_arr = np.array(lines)
    width_ = lines_arr[:, 2] - lines_arr[:, 0]
    height_ = lines_arr[:, 3] - lines_arr[:, 1]
    lines_index1 = np.where(width_ > 50)[0]
    lines_index2 = np.where(height_ > 50)[0]
    lines_index = np.hstack([lines_index1, lines_index2])
    new_lines = [lines[ele] for ele in lines_index]
    new_lines = clean_repeat_lines(new_lines)
    lines_tmp = filter_long_distance_lines(new_lines, sheet_dict)
    # lines_without_mark = filter_line_in_mark(new_lines, sheet_dict)

    # 检验滤线这一步骤有没有问题
    # for dline in lines_tmp:
    #     x0 = int(round(dline[0]))
    #     y0 = int(round(dline[1]))
    #     x1 = int(round(dline[2]))
    #     y1 = int(round(dline[3]))
    #     cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    #     text = str([x0, y0, x1, y1])
    #     cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\raw_lines.jpg', image)

    # get roi polygon 适当放大一点区域,使的线能包含在面里
    roi_box_polygon = []
    for index, region_box in enumerate(sheet_dict):
        if region_box['class_name'] in class_above_edge:
            coordinates = region_box['bounding_box']
            xmin = coordinates['xmin'] - 5
            ymin = coordinates['ymin'] - 5
            xmax = coordinates['xmax'] + 5
            ymax = coordinates['ymax'] + 5
            box_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            roi_box_polygon.append(box_polygon)
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\enlarge_box.jpg', image)

    # print(roi_box_polygon)

    # 对line进行分横向纵向
    landscape_list_tmp = []       # 横向
    longitudinal_list_tmp = []    # 纵向
    for ele in lines_tmp:
        if int(ele[2]) - int(ele[0]) < 10:     # 纵向
            longitudinal_list_tmp.append(ele)
        elif int(ele[3]) - int(ele[1]) < 10:   # 横向
            landscape_list_tmp.append(ele)
    print(longitudinal_list_tmp)


    points_all_list = []
    for index, box_polygon in enumerate(roi_box_polygon):

        points_list = []
        extend_line_lan = LineString()
        extend_line_lon = LineString()
        for line1 in landscape_list_tmp:
            for line2 in longitudinal_list_tmp:
                # 这部分对原先出来的线进行增大
                raw_line_lan = LineString([(line1[0], line1[1]), (line1[2], line1[3])])   # 横向增加5个像素
                line_start_lan, line_end_lan = raw_line_lan.bounds[0], raw_line_lan.bounds[1]

                raw_line_lon = LineString([(line2[0], line2[1]), (line2[2], line2[3])])          # 纵向的增加5个像素
                line_start_lon, line_end_lon = raw_line_lon.bounds[0], raw_line_lon.bounds[1]

                # decision1 = [raw_line_lan.within(box_polygon), raw_line_lan.contains(box_polygon), raw_line_lan.overlaps(box_polygon)]
                # decision2 = [raw_line_lon.within(box_polygon), raw_line_lon.contains(box_polygon), raw_line_lon.overlaps(box_polygon)]

                decision1 = [raw_line_lan.within(box_polygon), raw_line_lan.contains(box_polygon)]
                decision2 = [raw_line_lon.within(box_polygon), raw_line_lon.contains(box_polygon)]
                if True in decision1 and True in decision2:
                    # 这里需要考虑延长左边还是右边
                    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = box_polygon.bounds[0], box_polygon.bounds[1], \
                                                                 box_polygon.bounds[2], box_polygon.bounds[3]

                    # if abs(line_start_lan - bbox_xmin) < abs(bbox_xmax - line_end_lan):
                    extend_line_lan = LineString([(box_polygon.bounds[0], line1[1]), (box_polygon.bounds[2], line1[3])])

                    # elif abs(line_start_lon - bbox_ymin) < abs(bbox_ymax - line_end_lon):
                    extend_line_lon = LineString([(line2[0], box_polygon.bounds[1]), (line2[2], box_polygon.bounds[3])])

                    cond1 = extend_line_lan.intersects(extend_line_lon)  # T, L, 十交叉
                    cond2 = extend_line_lan.crosses(extend_line_lon)  # 十字交叉
                    cond3 = raw_line_lan.intersects(raw_line_lon)
                    cond4 = raw_line_lan.crosses(raw_line_lon)

                    xp, yp = 0.0, 0.0
                    if cond3:
                        (xp, yp) = raw_line_lan.intersection(raw_line_lon).bounds[:2]
                    elif cond1:
                        (xp, yp) = extend_line_lan.intersection(extend_line_lon).bounds[:2]
                    print((xp, yp))
                    points_list.append([(xp, yp), (raw_line_lan, raw_line_lon), (extend_line_lan, extend_line_lon)])
        # check points   判断点周围有没有黑色像素
        points_new = []
        for points_ in points_list:
            pixel_point = img.getpixel((int(points_[0][0]), int(points_[0][1])))
            if np.mean(np.array(pixel_point)) <= 250:
                points_new.append(points_)
        print(points_new)

        points_new = sorted(points_new, key=lambda k: k[0][0])
        points_new_list = []
        if len(points_new) == 1:
            points_new_list = points_new
        else:
            points_all = [ele[0][0] for ele in points_new]
            rear_one = np.array(points_all)[1:]
            former_one = np.array(points_all)[:-1]
            differ_x = rear_one - former_one
            index_x = np.where(differ_x >= 6.0)[0] + 1     # 符合条件的说明不是同一个点,这些点都要加进列表里去
            index_x = list(index_x)

            index_x.insert(0, 0)
            index_x.append(len(points_new))

            if len(index_x) > 2:
                for index_a, ele in enumerate(index_x[1:]):
                    block_one = points_new[index_x[index_a]:index_x[index_a + 1]]
                    if len(block_one) == 1:
                        points_new_list.extend(block_one)
                    else:
                        # 取均值
                        x_mean = np.mean(np.array([ele[0][0] for ele in block_one]))
                        y_mean = np.mean(np.array([ele[0][1] for ele in block_one]))
                        # 取最长的线

                        # max_index_lan = np.argsort(np.array([int(ele[1][0]) for ele in block_one]))
                        # max_index_lon = np.argsort(np.array([int(ele[1][1]) for ele in block_one]))

                        xx = [int(ele[1][0].length) for ele in block_one]
                        max_index_lan = np.argsort([int(ele[1][0].length) for ele in block_one])
                        yy = [int(ele[1][1].length) for ele in block_one]
                        max_index_lon = np.argsort([int(ele[1][1].length) for ele in block_one])
                        zz = block_one[max_index_lan[-1]][1][0]
                        cc = block_one[max_index_lon[-1]][1][1]
                        points_new_list.append([(x_mean, y_mean), (block_one[max_index_lan[-1]][1][0], block_one[max_index_lon[-1]][1][1])])

        print(points_new_list)

    # template = r'F:\exam\sheet_resolve\exam_info\000000-template.xml'
    # tree = ET.parse(template)
    # for ele in points_all_list:
    #     for points_ in ele:
    #         points1 = points_[1][0].bounds
    #         points2 = points_[1][1].bounds
    #         create_xml(str(points_[0]), tree, int(points1[0]), int(points1[1]), int(points1[2]), int(points1[3]))
    #         create_xml(str(points_[0]), tree, int(points2[0]), int(points2[1]), int(points2[2]), int(points2[3]))
    # tree.write(r'E:\December\math_12_18\1_18\test_img\save3\3.xml')


    #     # 合并points_new_list成一个list
    #     # points_new_list_tmp = [ele for ele in points_new_list]

        for points_and_lines in points_new_list:
            direction = get_direction(points_and_lines, box_polygon, image)
            if direction != None:
                points_dict = {}
                points_dict['points'] = points_and_lines[0]
                points_dict['direction'] = direction
                points_dict['extend_line'] = (extend_line_lan, extend_line_lon)
                points_all_list.append(points_dict)
        print(points_all_list)

    template = r'F:\exam\sheet_resolve\exam_info\000000-template.xml'
    tree = ET.parse(template)
    for points_ in points_all_list:
        points = points_['points']
        create_xml(str(points) + '_' + points_['direction'], tree, int(points[0]), int(points[1]), int(points[0] + 1), int(points[1] + 1))
    tree.write(r'E:\December\math_12_18\1_18\test_img\save3\3.xml')


if __name__ == '__main__':
    img_path = r'E:\December\math_12_18\1_18\test_img\3.jpg'
    # sheet_dict = [{'class_name': 'alarm_info', 'bounding_box': {'xmin': 2312, 'ymin': 158, 'xmax': 3096, 'ymax': 182, 'xmid': 2704, 'ymid': 170}, 'score': '1.0000'}, {'class_name': 'alarm_info', 'bounding_box': {'xmin': 1378, 'ymin': 158, 'xmax': 2161, 'ymax': 184, 'xmid': 1769, 'ymid': 171}, 'score': '1.0000'}, {'class_name': 'alarm_info', 'bounding_box': {'xmin': 2313, 'ymin': 2072, 'xmax': 3097, 'ymax': 2097, 'xmid': 2705, 'ymid': 2084}, 'score': '0.9999'}, {'class_name': 'alarm_info', 'bounding_box': {'xmin': 1380, 'ymin': 2072, 'xmax': 2163, 'ymax': 2098, 'xmid': 1771, 'ymid': 2085}, 'score': '0.8631'}, {'class_name': 'alarm_info', 'bounding_box': {'xmin': 448, 'ymin': 2075, 'xmax': 1228, 'ymax': 2101, 'xmid': 838, 'ymid': 2088}, 'score': '0.8080'}, {'class_name': 'attention', 'bounding_box': {'xmin': 370, 'ymin': 406, 'xmax': 1314, 'ymax': 574}, 'score': '1.0000'}, {'class_name': 'bar_code', 'bounding_box': {'xmin': 805, 'ymin': 249, 'xmax': 1288, 'ymax': 414, 'xmid': 1046, 'ymid': 331}, 'score': '1.0000'}, {'class_name': 'choice', 'bounding_box': {'xmin': 1325, 'ymin': 1248, 'xmax': 2215, 'ymax': 1531}, 'score': '0.9209'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 774, 'ymin': 733, 'xmax': 934, 'ymax': 764, 'xmid': 854, 'ymid': 748}, 'score': '0.9992'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 1703, 'ymin': 1406, 'xmax': 1865, 'ymax': 1430, 'xmid': 1784, 'ymid': 1418}, 'score': '0.9928'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 1011, 'ymin': 733, 'xmax': 1172, 'ymax': 762, 'xmid': 1091, 'ymid': 747}, 'score': '0.9898'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 2863, 'ymin': 834, 'xmax': 3026, 'ymax': 860, 'xmid': 2944, 'ymid': 847}, 'score': '0.9841'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 2415, 'ymin': 833, 'xmax': 2579, 'ymax': 859, 'xmid': 2497, 'ymid': 846}, 'score': '0.9322'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 1666, 'ymin': 1391, 'xmax': 1703, 'ymax': 1440}, 'score': '1.0000'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 1960, 'ymin': 816, 'xmax': 1999, 'ymax': 855}, 'score': '1.0000'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 1756, 'ymin': 815, 'xmax': 1795, 'ymax': 855}, 'score': '1.0000'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 717, 'ymin': 1538, 'xmax': 751, 'ymax': 1579}, 'score': '1.0000'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 725, 'ymin': 860, 'xmax': 757, 'ymax': 901}, 'score': '1.0000'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 985, 'ymin': 860, 'xmax': 1017, 'ymax': 901}, 'score': '1.0000'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 1551, 'ymin': 811, 'xmax': 1592, 'ymax': 852}, 'score': '1.0000'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 516, 'ymin': 723, 'xmax': 548, 'ymax': 767}, 'score': '0.9999'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 2832, 'ymin': 824, 'xmax': 2870, 'ymax': 870}, 'score': '0.9999'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 2603, 'ymin': 823, 'xmax': 2641, 'ymax': 867}, 'score': '0.9999'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 981, 'ymin': 726, 'xmax': 1016, 'ymax': 771}, 'score': '0.9999'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 739, 'ymin': 728, 'xmax': 772, 'ymax': 772}, 'score': '0.9999'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 2378, 'ymin': 823, 'xmax': 2417, 'ymax': 867}, 'score': '0.9997'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 759, 'ymin': 865, 'xmax': 914, 'ymax': 896}, 'score': '0.9980'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 748, 'ymin': 1540, 'xmax': 911, 'ymax': 1571}, 'score': '0.9360'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 547, 'ymin': 733, 'xmax': 712, 'ymax': 762}, 'score': '0.9064'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 1016, 'ymin': 864, 'xmax': 1174, 'ymax': 893}, 'score': '0.7220'}, {'class_name': 'exam_number_w', 'bounding_box': {'xmin': 359, 'ymin': 318, 'xmax': 805, 'ymax': 377}, 'score': '1.0000'}, {'class_name': 'info_title', 'bounding_box': {'xmin': 503, 'ymin': 142, 'xmax': 1210, 'ymax': 258, 'xmid': 856, 'ymid': 200}, 'score': '1.0000'}, {'class_name': 'lack', 'bounding_box': {'xmin': 382, 'ymin': 565, 'xmax': 1308, 'ymax': 620}, 'score': '0.9997'}, {'class_name': 'line_0', 'bounding_box': {'xmin': 1, 'ymin': 1, 'xmax': 3, 'ymax': 2216}, 'score': '0.8634'}, {'class_name': 'score_collect', 'bounding_box': {'xmin': 2273, 'ymin': 705, 'xmax': 2500, 'ymax': 820}, 'score': '1.0000'}, {'class_name': 'score_collect', 'bounding_box': {'xmin': 401, 'ymin': 1499, 'xmax': 617, 'ymax': 1613}, 'score': '1.0000'}, {'class_name': 'score_collect', 'bounding_box': {'xmin': 2268, 'ymin': 217, 'xmax': 2499, 'ymax': 329}, 'score': '1.0000'}, {'class_name': 'score_collect', 'bounding_box': {'xmin': 1328, 'ymin': 772, 'xmax': 1558, 'ymax': 886}, 'score': '1.0000'}, {'class_name': 'score_collect', 'bounding_box': {'xmin': 1332, 'ymin': 1313, 'xmax': 1558, 'ymax': 1427}, 'score': '1.0000'}, {'class_name': 'score_collect', 'bounding_box': {'xmin': 398, 'ymin': 823, 'xmax': 614, 'ymax': 939}, 'score': '1.0000'}, {'class_name': 'seal_area', 'bounding_box': {'xmin': 4, 'ymin': 140, 'xmax': 297, 'ymax': 2169}, 'score': '1.0000'}, {'class_name': 'solve', 'bounding_box': {'xmin': 1325, 'ymin': 713, 'xmax': 2215, 'ymax': 1246, 'xmid': 1770, 'ymid': 979}, 'score': '1.0000'}, {'class_name': 'solve', 'bounding_box': {'xmin': 2258, 'ymin': 193, 'xmax': 3155, 'ymax': 633, 'xmid': 2706, 'ymid': 413}, 'score': '1.0000'}, {'class_name': 'solve', 'bounding_box': {'xmin': 2258, 'ymin': 1207, 'xmax': 3154, 'ymax': 2062, 'xmid': 2706, 'ymid': 1634}, 'score': '1.0000'}, {'class_name': 'solve', 'bounding_box': {'xmin': 390, 'ymin': 808, 'xmax': 1284, 'ymax': 1469, 'xmid': 837, 'ymid': 1138}, 'score': '1.0000'}, {'class_name': 'solve', 'bounding_box': {'xmin': 1325, 'ymin': 194, 'xmax': 2215, 'ymax': 711, 'xmid': 1770, 'ymid': 452}, 'score': '0.9999'}, {'class_name': 'solve', 'bounding_box': {'xmin': 2258, 'ymin': 635, 'xmax': 3155, 'ymax': 1204, 'xmid': 2706, 'ymid': 919}, 'score': '0.9999'}, {'class_name': 'solve', 'bounding_box': {'xmin': 390, 'ymin': 1471, 'xmax': 1285, 'ymax': 2065, 'xmid': 837, 'ymid': 1768}, 'score': '0.9999'}, {'class_name': 'solve', 'bounding_box': {'xmin': 1325, 'ymin': 1533, 'xmax': 2215, 'ymax': 2063, 'xmid': 1770, 'ymid': 1798}, 'score': '0.9991'}, {'class_name': 'solve_without_type_score', 'bounding_box': {'xmin': 1305, 'ymin': 1259, 'xmax': 2244, 'ymax': 1530}, 'score': '0.9900'}, {'class_name': 'student_info', 'bounding_box': {'xmin': 27, 'ymin': 1684, 'xmax': 205, 'ymax': 2105}, 'score': '1.0000'}, {'class_name': 'student_info', 'bounding_box': {'xmin': 26, 'ymin': 927, 'xmax': 205, 'ymax': 1305}, 'score': '1.0000'}, {'class_name': 'student_info', 'bounding_box': {'xmin': 31, 'ymin': 403, 'xmax': 215, 'ymax': 916}, 'score': '1.0000'}, {'class_name': 'student_info', 'bounding_box': {'xmin': 29, 'ymin': 1305, 'xmax': 202, 'ymax': 1672}, 'score': '1.0000'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1362, 'ymin': 737, 'xmax': 1620, 'ymax': 763}, 'score': '1.0000'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1354, 'ymin': 265, 'xmax': 1440, 'ymax': 288}, 'score': '1.0000'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 2299, 'ymin': 1250, 'xmax': 2399, 'ymax': 1273}, 'score': '1.0000'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1351, 'ymin': 1565, 'xmax': 1458, 'ymax': 1589}, 'score': '1.0000'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 413, 'ymin': 663, 'xmax': 590, 'ymax': 688}, 'score': '1.0000'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 2299, 'ymin': 879, 'xmax': 2398, 'ymax': 902}, 'score': '1.0000'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 2308, 'ymin': 669, 'xmax': 2513, 'ymax': 694}, 'score': '1.0000'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 2518, 'ymin': 296, 'xmax': 2619, 'ymax': 319}, 'score': '0.9999'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 2511, 'ymin': 246, 'xmax': 2762, 'ymax': 271}, 'score': '0.9985'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1339, 'ymin': 725, 'xmax': 1381, 'ymax': 777}, 'score': '1.0000'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 385, 'ymin': 651, 'xmax': 432, 'ymax': 697}, 'score': '1.0000'}, {'class_name': 'w_h_blank', 'bounding_box': {'xmin': 1, 'ymin': 2106, 'xmax': 1304, 'ymax': 2262}}, {'class_name': 'w_h_blank', 'bounding_box': {'xmin': 1305, 'ymin': 2103, 'xmax': 2235, 'ymax': 2262}}, {'class_name': 'w_h_blank', 'bounding_box': {'xmin': 2236, 'ymin': 2102, 'xmax': 3266, 'ymax': 2262}}]
    sheet_dict = [{'class_name': 'bar_code',
                     'bounding_box': {'xmin': 1215, 'ymin': 770, 'xmax': 1598, 'ymax': 1433, 'xmid': 1406,
                                      'ymid': 1101}, 'score': '0.9493'}, {'class_name': 'attention',
                                                                          'bounding_box': {'xmin': 179, 'ymin': 440,
                                                                                           'xmax': 1092, 'ymax': 729,
                                                                                           'xmid': 635, 'ymid': 584},
                                                                          'score': '0.9449'}, {'class_name': 'solve',
                                                                                               'bounding_box': {
                                                                                                   'xmin': 1815,
                                                                                                   'ymin': 149,
                                                                                                   'xmax': 3163,
                                                                                                   'ymax': 1075,
                                                                                                   'xmid': 2489,
                                                                                                   'ymid': 612},
                                                                                               'score': '0.9369'},
                    {'class_name': 'solve',
                     'bounding_box': {'xmin': 1811, 'ymin': 1205, 'xmax': 3158, 'ymax': 3239, 'xmid': 2484,
                                      'ymid': 2222}, 'score': '0.9278'}, {'class_name': 'solve',
                                                                          'bounding_box': {'xmin': 3344, 'ymin': 229,
                                                                                           'xmax': 4691, 'ymax': 3240,
                                                                                           'xmid': 4017, 'ymid': 1734},
                                                                          'score': '0.9262'}, {'class_name': 'mark',
                                                                                               'bounding_box': {
                                                                                                   'xmin': 256,
                                                                                                   'ymin': 2920,
                                                                                                   'xmax': 1631,
                                                                                                   'ymax': 3006,
                                                                                                   'xmid': 943,
                                                                                                   'ymid': 2963},
                                                                                               'score': '0.9176'},
                    {'class_name': 'mark',
                     'bounding_box': {'xmin': 1800, 'ymin': 1129, 'xmax': 3166, 'ymax': 1214, 'xmid': 2483,
                                      'ymid': 1171}, 'score': '0.9120'}, {'class_name': 'choice',
                                                                          'bounding_box': {'xmin': 328, 'ymin': 1643,
                                                                                           'xmax': 1589, 'ymax': 2004,
                                                                                           'xmid': 958, 'ymid': 1823},
                                                                          'score': '0.9108'}, {'class_name': 'choice_m',
                                                                                               'bounding_box': {
                                                                                                   'xmin': 1288,
                                                                                                   'ymin': 1648,
                                                                                                   'xmax': 1607,
                                                                                                   'ymax': 1979,
                                                                                                   'xmid': 1447,
                                                                                                   'ymid': 1813},
                                                                                               'score': '0.9021'},
                    {'class_name': 'choice_s',
                     'bounding_box': {'xmin': 841, 'ymin': 1678, 'xmax': 1127, 'ymax': 1723, 'xmid': 984, 'ymid': 1700},
                     'score': '0.8984'},
                    {'class_name': 'choice_n', 'bounding_box': {'xmin': 1212, 'ymin': 1668, 'xmax': 1295, 'ymax': 1998},
                     'score': '0.8955', 'numbers': [{'digital': 11, 'loc': (1232, 1845, 1276, 1891, 1254, 1868)},
                                                    {'digital': 12, 'loc': (1235, 1932, 1271, 1977, 1253, 1954)}]},
                    {'class_name': 'choice_s',
                     'bounding_box': {'xmin': 836, 'ymin': 1925, 'xmax': 1127, 'ymax': 1973, 'xmid': 981, 'ymid': 1949},
                     'score': '0.8930'}, {'class_name': 'choice_s',
                                          'bounding_box': {'xmin': 1298, 'ymin': 1679, 'xmax': 1582, 'ymax': 1724,
                                                           'xmid': 1440, 'ymid': 1701}, 'score': '0.8920'},
                    {'class_name': 'type_score', 'bounding_box': {'xmin': 280, 'ymin': 2998, 'xmax': 848, 'ymax': 3064},
                     'score': '0.8916'}, {'class_name': 'cloze_s',
                                          'bounding_box': {'xmin': 267, 'ymin': 2566, 'xmax': 1600, 'ymax': 2687,
                                                           'xmid': 933, 'ymid': 2626}, 'score': '0.8901'},
                    {'class_name': 'choice_s',
                     'bounding_box': {'xmin': 1306, 'ymin': 1933, 'xmax': 1567, 'ymax': 1974, 'xmid': 1436,
                                      'ymid': 1953}, 'score': '0.8897'}, {'class_name': 'choice_s',
                                                                          'bounding_box': {'xmin': 384, 'ymin': 1670,
                                                                                           'xmax': 678, 'ymax': 1723,
                                                                                           'xmid': 531, 'ymid': 1696},
                                                                          'score': '0.8893'},
                    {'class_name': 'type_score',
                     'bounding_box': {'xmin': 146, 'ymin': 1465, 'xmax': 1760, 'ymax': 1615}, 'score': '0.8884'},
                    {'class_name': 'info_title',
                     'bounding_box': {'xmin': 427, 'ymin': 133, 'xmax': 1496, 'ymax': 378, 'xmid': 961, 'ymid': 255},
                     'score': '0.8852'}, {'class_name': 'choice_s',
                                          'bounding_box': {'xmin': 1298, 'ymin': 1765, 'xmax': 1581, 'ymax': 1809,
                                                           'xmid': 1439, 'ymid': 1787}, 'score': '0.8850'},
                    {'class_name': 'choice_s',
                     'bounding_box': {'xmin': 388, 'ymin': 1841, 'xmax': 674, 'ymax': 1888, 'xmid': 531, 'ymid': 1864},
                     'score': '0.8822'}, {'class_name': 'solve',
                                          'bounding_box': {'xmin': 274, 'ymin': 2926, 'xmax': 1619, 'ymax': 3268,
                                                           'xmid': 946, 'ymid': 3097}, 'score': '0.8799'},
                    {'class_name': 'mark',
                     'bounding_box': {'xmin': 3330, 'ymin': 150, 'xmax': 4704, 'ymax': 237, 'xmid': 4017, 'ymid': 193},
                     'score': '0.8788'}, {'class_name': 'cloze_score',
                                          'bounding_box': {'xmin': 1244, 'ymin': 2467, 'xmax': 1597, 'ymax': 2563},
                                          'score': '0.8782'}, {'class_name': 'cloze_score',
                                                               'bounding_box': {'xmin': 1247, 'ymin': 2249,
                                                                                'xmax': 1594, 'ymax': 2348},
                                                               'score': '0.8781'},
                    {'class_name': 'qr_code', 'bounding_box': {'xmin': 1287, 'ymin': 480, 'xmax': 1536, 'ymax': 719},
                     'score': '0.8780'}, {'class_name': 'choice_s',
                                          'bounding_box': {'xmin': 841, 'ymin': 1842, 'xmax': 1124, 'ymax': 1890,
                                                           'xmid': 982, 'ymid': 1866}, 'score': '0.8777'},
                    {'class_name': 'choice_s',
                     'bounding_box': {'xmin': 839, 'ymin': 1762, 'xmax': 1131, 'ymax': 1809, 'xmid': 985, 'ymid': 1785},
                     'score': '0.8773'}, {'class_name': 'cloze_score',
                                          'bounding_box': {'xmin': 1248, 'ymin': 2358, 'xmax': 1596, 'ymax': 2455},
                                          'score': '0.8761'}, {'class_name': 'choice_s',
                                                               'bounding_box': {'xmin': 1295, 'ymin': 1848,
                                                                                'xmax': 1585, 'ymax': 1891,
                                                                                'xmid': 1440, 'ymid': 1869},
                                                               'score': '0.8759'}, {'class_name': 'exam_number_s',
                                                                                    'bounding_box': {'xmin': 440,
                                                                                                     'ymin': 839,
                                                                                                     'xmax': 500,
                                                                                                     'ymax': 1427},
                                                                                    'score': '0.8746'},
                    {'class_name': 'exam_number_s',
                     'bounding_box': {'xmin': 634, 'ymin': 826, 'xmax': 694, 'ymax': 1438}, 'score': '0.8675'},
                    {'class_name': 'cloze_score',
                     'bounding_box': {'xmin': 1239, 'ymin': 2577, 'xmax': 1606, 'ymax': 2672}, 'score': '0.8640'},
                    {'class_name': 'type_score', 'bounding_box': {'xmin': 3360, 'ymin': 236, 'xmax': 3934, 'ymax': 295},
                     'score': '0.8612'},
                    {'class_name': 'choice_n', 'bounding_box': {'xmin': 769, 'ymin': 1656, 'xmax': 825, 'ymax': 1999},
                     'score': '0.8609', 'numbers': [{'digital': 7, 'loc': (788, 1845, 810, 1891, 799, 1868)},
                                                    {'digital': 8, 'loc': (787, 1932, 809, 1977, 798, 1954)}]},
                    {'class_name': 'type_score_n',
                     'bounding_box': {'xmin': 281, 'ymin': 2444, 'xmax': 377, 'ymax': 2555}, 'score': '0.8598'},
                    {'class_name': 'exam_number_s',
                     'bounding_box': {'xmin': 310, 'ymin': 824, 'xmax': 368, 'ymax': 1422}, 'score': '0.8596'},
                    {'class_name': 'type_score',
                     'bounding_box': {'xmin': 203, 'ymin': 2729, 'xmax': 1648, 'ymax': 2869}, 'score': '0.8595'},
                    {'class_name': 'choice_s',
                     'bounding_box': {'xmin': 390, 'ymin': 1925, 'xmax': 669, 'ymax': 1971, 'xmid': 529, 'ymid': 1948},
                     'score': '0.8588'}, {'class_name': 'type_score',
                                          'bounding_box': {'xmin': 1808, 'ymin': 1206, 'xmax': 2437, 'ymax': 1285},
                                          'score': '0.8539'}, {'class_name': 'cloze_s',
                                                               'bounding_box': {'xmin': 271, 'ymin': 2457, 'xmax': 1600,
                                                                                'ymax': 2570, 'xmid': 935,
                                                                                'ymid': 2513}, 'score': '0.8497'},
                    {'class_name': 'choice_s',
                     'bounding_box': {'xmin': 386, 'ymin': 1758, 'xmax': 678, 'ymax': 1807, 'xmid': 532, 'ymid': 1782},
                     'score': '0.8479'}, {'class_name': 'type_score_n',
                                          'bounding_box': {'xmin': 278, 'ymin': 2343, 'xmax': 376, 'ymax': 2454},
                                          'score': '0.8368'}, {'class_name': 'cloze_s',
                                                               'bounding_box': {'xmin': 265, 'ymin': 2219, 'xmax': 1597,
                                                                                'ymax': 2356, 'xmid': 931,
                                                                                'ymid': 2287}, 'score': '0.8358'},
                    {'class_name': 'type_score_n',
                     'bounding_box': {'xmin': 275, 'ymin': 2231, 'xmax': 376, 'ymax': 2345}, 'score': '0.8348'},
                    {'class_name': 'type_score_n',
                     'bounding_box': {'xmin': 1816, 'ymin': 1210, 'xmax': 1911, 'ymax': 1291}, 'score': '0.8314'},
                    {'class_name': 'choice_n', 'bounding_box': {'xmin': 318, 'ymin': 1656, 'xmax': 370, 'ymax': 1988},
                     'score': '0.8236'}, {'class_name': 'type_score_n',
                                          'bounding_box': {'xmin': 272, 'ymin': 2996, 'xmax': 359, 'ymax': 3074},
                                          'score': '0.8213'}, {'class_name': 'type_score_n',
                                                               'bounding_box': {'xmin': 172, 'ymin': 2718, 'xmax': 273,
                                                                                'ymax': 2800}, 'score': '0.8206'},
                    {'class_name': 'type_score',
                     'bounding_box': {'xmin': 154, 'ymin': 2023, 'xmax': 1748, 'ymax': 2178}, 'score': '0.8109'},
                    {'class_name': 'type_score_n',
                     'bounding_box': {'xmin': 277, 'ymin': 2557, 'xmax': 369, 'ymax': 2664}, 'score': '0.8105'},
                    {'class_name': 'page',
                     'bounding_box': {'xmin': 865, 'ymin': 3382, 'xmax': 1101, 'ymax': 3413, 'xmid': 984, 'ymid': 3398},
                     'score': '0.7981'}, {'class_name': 'type_score_n',
                                          'bounding_box': {'xmin': 181, 'ymin': 1452, 'xmax': 284, 'ymax': 1547},
                                          'score': '0.7877'}, {'class_name': 'exam_number_s',
                                                               'bounding_box': {'xmin': 506, 'ymin': 849, 'xmax': 566,
                                                                                'ymax': 1430}, 'score': '0.7606'},
                    {'class_name': 'page',
                     'bounding_box': {'xmin': 2426, 'ymin': 3388, 'xmax': 2663, 'ymax': 3420, 'xmid': 2545,
                                      'ymid': 3404}, 'score': '0.7590'}, {'class_name': 'type_score_n',
                                                                          'bounding_box': {'xmin': 170, 'ymin': 2020,
                                                                                           'xmax': 283, 'ymax': 2110},
                                                                          'score': '0.7576'},
                    {'class_name': 'exam_number_s',
                     'bounding_box': {'xmin': 697, 'ymin': 822, 'xmax': 758, 'ymax': 1449}, 'score': '0.7533'},
                    {'class_name': 'exam_number_s',
                     'bounding_box': {'xmin': 570, 'ymin': 834, 'xmax': 630, 'ymax': 1443}, 'score': '0.7221'},
                    {'class_name': 'type_score_n',
                     'bounding_box': {'xmin': 3356, 'ymin': 229, 'xmax': 3435, 'ymax': 301}, 'score': '0.7016'},
                    {'class_name': 'cloze_s',
                     'bounding_box': {'xmin': 268, 'ymin': 2350, 'xmax': 1606, 'ymax': 2458, 'xmid': 937, 'ymid': 2404},
                     'score': '0.7002'}, {'class_name': 'exam_number',
                                          'bounding_box': {'xmin': 292, 'ymin': 822, 'xmax': 836, 'ymax': 1449,
                                                           'xmid': 564, 'ymid': 1135}},
                    {'class_name': 'choice_m', 'number': [-1],
                     'bounding_box': {'xmin': 373, 'ymin': 1656, 'xmax': 696, 'ymax': 1988, 'xmid': 534, 'ymid': 1822},
                     'choice_option': 'A,B,C,D', 'default_points': [5], 'direction': 180, 'cols': 4, 'rows': 1},
                    {'class_name': 'choice_m', 'number': [7, 8],
                     'bounding_box': {'xmin': 828, 'ymin': 1656, 'xmax': 1149, 'ymax': 1999, 'xmid': 988, 'ymid': 1827},
                     'choice_option': 'A,B,C,D', 'default_points': [5, 5], 'direction': 180, 'cols': 4, 'rows': 2},
                    {'class_name': 'cloze',
                     'bounding_box': {'xmin': 11, 'ymin': 2234, 'xmax': 1714, 'ymax': 2656, 'xmid': 862, 'ymid': 2445}},
                    {'class_name': 'solve_without_type_score',
                     'bounding_box': {'xmin': 1, 'ymin': 2691, 'xmax': 1714, 'ymax': 2917}},
                    {'class_name': 'w_h_blank', 'bounding_box': {'xmin': 1, 'ymin': 3272, 'xmax': 1714, 'ymax': 3379}},
                    {'class_name': 'w_h_blank', 'bounding_box': {'xmin': 1, 'ymin': 3417, 'xmax': 1714, 'ymax': 3418}},
                    {'class_name': 'w_h_blank', 'bounding_box': {'xmin': 1715, 'ymin': 3, 'xmax': 3252, 'ymax': 146}},
                    {'class_name': 'w_h_blank',
                     'bounding_box': {'xmin': 1715, 'ymin': 1079, 'xmax': 3252, 'ymax': 1126}},
                    {'class_name': 'w_h_blank',
                     'bounding_box': {'xmin': 1715, 'ymin': 3243, 'xmax': 3252, 'ymax': 3385}},
                    {'class_name': 'w_h_blank', 'bounding_box': {'xmin': 3253, 'ymin': 3, 'xmax': 4952, 'ymax': 147}},
                    {'class_name': 'w_h_blank',
                     'bounding_box': {'xmin': 3253, 'ymin': 3244, 'xmax': 4952, 'ymax': 3417}}]
    split_x = [1715, 3253]
    image = cv2.imread(img_path)
    fld_demo4(image, sheet_dict, split_x)