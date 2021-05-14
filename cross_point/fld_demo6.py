# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/4/26 11:23'


import cv2, time, heapq, os, copy, shutil
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
    # cv2.imwrite(os.path.join(r'E:\December\math_12_18\1_18\test_img\save3\\', 'THRESH_BINARY_INV' + '.jpg'), binary)

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

    # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\extend_lines_by_bbox.jpg', image)

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
        # cv2.imwrite(r'E:\December\math_12_18\1_18\test_img\save3\mark.jpg', image)
    return lines_tmp


def fld_demo4(image, sheet_dict, name, save_path):
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
    for dline in lines_tmp:
        x0 = int(round(dline[0]))
        y0 = int(round(dline[1]))
        x1 = int(round(dline[2]))
        y1 = int(round(dline[3]))
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
        text = str([x0, y0, x1, y1])
        cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    write_single_img(image, os.path.join(save_path, name + '_raw_lines' + '.jpg'))

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
    # print(longitudinal_list_tmp)


    points_all_list = []
    # points_list = []

    for index, box_polygon in enumerate(roi_box_polygon):
        extend_line_lan = LineString()
        extend_line_lon = LineString()
        points_list = []
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

                    cond1 = raw_line_lan.intersects(raw_line_lon)
                    cond2 = raw_line_lan.crosses(raw_line_lon)

                    cond3 = extend_line_lan.intersects(extend_line_lon)  # 十字交叉
                    cond4 = extend_line_lan.crosses(extend_line_lon)  # 十字交叉

                    xp, yp = 0.0, 0.0
                    if cond1:
                        (xp, yp) = raw_line_lan.intersection(raw_line_lon).bounds[:2]
                    elif cond3:
                        (xp, yp) = extend_line_lan.intersection(extend_line_lon).bounds[:2]
                    points_list.append([(xp, yp), (raw_line_lan, raw_line_lon), (extend_line_lan, extend_line_lon)])

    # TODO 验证找所有点的正确性
    # template = r'F:\exam\sheet_resolve\exam_info\000000-template.xml'
    # tree = ET.parse(template)
    # for ele in points_list:
    #     # for points2 in ele:
    #         # points1 = points_[1][0].bounds
    #         # points2 = points_[1][1].bounds
    #         # create_xml(str(points_[0]), tree, int(points1[0]), int(points1[1]), int(points1[2]), int(points1[3]))
    #         # create_xml(str(points_[0]), tree, int(points2[0]), int(points2[1]), int(points2[2]), int(points2[3]))
    #     points2 = ele[0]
    #     create_xml('points', tree, int(points2[0]), int(points2[1]), int(points2[0] + 1), int(points2[1] + 1))
    # tree.write(os.path.join(save_path, name + '.xml'))


            # 找出角点最优的那个
            # check points   判断点周围有没有黑色像素
        print('points_list:', points_list)
        points_new = []
        for points_ in points_list:
            pixel_point = img.getpixel((int(points_[0][0]), int(points_[0][1])))
            if int(np.mean(np.array(pixel_point))) <= 245:
                points_new.append(points_)
        # print(points_new)

        # 验证这里每个点是不是黑色像素
        template = r'F:\exam\sheet_resolve\exam_info\000000-template.xml'
        tree = ET.parse(template)
        for ele in points_list:
            points2 = ele[0]
            create_xml('points', tree, int(points2[0]), int(points2[1]), int(points2[0] + 1), int(points2[1] + 1))
        tree.write(os.path.join(save_path, 'points_' + name + '.xml'))

        points_new = sorted(points_new, key=lambda k: k[0][0])
        points_new_list = []
        if len(points_new) == 1:
            points_new_list.extend(points_new)
        else:
            points_all_x = [ele[0][0] for ele in points_new]
            rear_one_x = np.array(points_all_x)[1:]
            former_one_x = np.array(points_all_x)[:-1]
            differ_x = rear_one_x - former_one_x
            index_x = list(np.where(differ_x >= 6.0)[0] + 1)     # 符合条件的说明不是同一个点,这些点都要加进列表里去

            # 头和尾加进去
            # if points_all_x[1] - points_all_x[0] >= 6.0:
            #     index_x.insert(0, 0)
            # elif points_all_x[-1] - points_all_x[-1] >= 6.0:
            #     index_x.append(len(points_all_x) - 1)

            points_all_y = [ele[0][1] for ele in points_new]
            rear_one_y = np.array(points_all_y)[1:]
            former_one_y = np.array(points_all_y)[:-1]
            differ_y = rear_one_y - former_one_y
            index_y = list(np.where(differ_y >= 6.0)[0] + 1)     # 符合条件的说明不是同一个点,这些点都要加进列表里去

            # 头和尾加进去
            # if points_all_y[1] - points_all_y[0] >= 6.0:
            #     index_y.insert(0, 0)
            # elif points_all_y[-1] - points_all_y[-1] >= 6.0:
            #     index_y.append(len(points_all_y) - 1)

            if len(index_x) == 0 and len(index_y) == 0:   # 说明每个点都不重合,都是需要的点
                points_new_list.extend(points_new)
                for points_and_lines in points_new_list:
                    direction = get_direction(points_and_lines, box_polygon, image)
                    if direction != None:
                        points_dict = {}
                        points_dict['points'] = points_and_lines[0]
                        points_dict['direction'] = direction
                        points_dict['extend_line'] = (extend_line_lan, extend_line_lon)
                        points_all_list.append(points_dict)
            elif len(index_x) != 0 and len(index_y) == 0:
                if len(index_x) > 2:
                    index_x = list(index_x)
                    index_x.insert(0, 0)
                    index_x.append(len(points_new))

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
                print('points_new_list:', points_new_list)
            elif len(index_x) == 0 and len(index_y) != 0:
                index_y = list(index_y)

                index_y.insert(0, 0)
                index_y.append(len(points_new))

                if len(index_y) > 2:
                    for index_a, ele in enumerate(index_y[1:]):
                        block_one = points_new[index_y[index_a]:index_y[index_a + 1]]
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
                            points_new_list.append([(x_mean, y_mean), (
                            block_one[max_index_lan[-1]][1][0], block_one[max_index_lon[-1]][1][1])])
                    print('points_new_list:', points_new_list)
                else:
                    points_all_list.extend(points_new)

                # 合并points_new_list成一个list
                # points_new_list_tmp = [ele for ele in points_new_list]

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
    tree.write(os.path.join(save_path, name + '.xml'))


if __name__ == '__main__':
    save_path = r'E:\111_4_26_test_img\aa\save'
    img_path = r'E:\111_4_26_test_img\aa\22'
    img_list = os.listdir(img_path)
    for ele in img_list:
        if ele.endswith('.jpg'):
            img_path0 = os.path.join(img_path, ele)
            shutil.copyfile(img_path0, os.path.join(save_path, ele))
            xml_path0 = img_path0.replace('.jpg', '.xml')
            regions = read_xml_to_json(xml_path0)
            sheet_dict = regions['regions']
            image = read_single_img(img_path0)
            name = ele.split('.')[0]
            fld_demo4(image, sheet_dict, name, save_path)