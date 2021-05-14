# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/4/15 20:02'


import cv2, time
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point


def FLD_detect_raw(image):
    start_time = time.time()
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(img_gray)
    end_time = time.time()
    cost_time = end_time - start_time
    print('cost_time:', cost_time)
    for dline in dlines:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imshow("FLD", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(r'E:\December\chinese\unblank\save.jpg', image)


def clean_repeat_lines_(lines):
    lines_temp = lines.copy()
    lines_temp_arr = np.array(lines_temp)
    index1 = np.where(lines_temp_arr[0] < 0)[0]
    if index1:
        index1_list = index1.tolist()
        for i in index1_list:
            lines_temp_arr[i][0] = -lines_temp_arr[0]   # rho
            lines_temp_arr[i][1] = lines_temp_arr[1] - np.pi   # theta

    # 弧度小于10,角度小于0.1
    # rho = lines_temp_arr[:, 0]
    # differ_rho = rho[1:] - rho[:-1]
    # index_rho = np.where(differ_rho < 10)[0] + 1
    # theta = lines_temp_arr[:, 1]
    # differ_theta = theta[1:] - theta[:-1]
    # index_theta = np.where(differ_theta < 0.1)[0] + 1
    # print(theta)
    # common_index = list(set(index_rho).intersection(set(index_theta)))
    # newlines_tmp = [lines[ele] for ele in common_index]
    # print(newlines_tmp)

    newlines = []
    newlines.append(lines.pop(5))
    for line in lines:
        flag = 0
        for newline in newlines:
            if((abs(line[0]-newline[0])<10) & (abs(line[1]-newline[1]) < 0.1)):
                flag = 1
        if(flag == 0):
            newlines.append(line)
    print(newlines)
    return newlines


def cross_point1(line1, line2, image):
    point_is_exist = False
    x = y = 0
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True

    if k1 > 0 and k2 > 0:
        direction = 'right top'
    elif k1 > 0 and k2 < 0:
        direction = 'left top'
    elif k1 < 0 and k2 > 0:
        direction = 'left bottom'
    elif k1 < 0 and k2 < 0:
        direction = 'right bottom'
    if point_is_exist:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(image, (int(x3), int(y3)), (int(x4), int(y4)), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(r'E:\December\chinese\unblank\save2.jpg', image)
        return direction, [x, y]


# 计算两条线是否相交
def iscrosses(line1, line2):
    if LineString(line1).crosses(LineString(line2)):
        return True
    return False


def get_intersection_points_raw(lines_tmp, image):
    direction = {0: 0, 1: 90, 2: 180, 3: 270}
    lines = lines_tmp.copy()
    points = []
    # if (len(lines) == 4):
    horLine = []   # 横线
    verLine = []   # 竖线
    for line in lines:
        if int(line[3]) - int(line[1]) > 0 and int(line[2]) - int(line[0]) > 0:
            if int(line[3]) - int(line[1]) > 10:
                horLine.append(line)
                cv2.rectangle(image, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 255, 0), 2)
            else:
                verLine.append(line)
                cv2.rectangle(image, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 0, 255), 2)
    for l1 in horLine:
        for l2 in verLine:
            a = np.array([
                [np.cos(l1[1]), np.sin(l1[1])],
                [np.cos(l2[1]), np.sin(l2[1])]
            ])
            b = np.array([l1[0], l2[0]])
            points.append(np.linalg.solve(a, b))
        return points
    else:
        print("the number of lines error")


def get_intersection_points(lines, image):
    lines_tmp = [line for line in lines if int(line[3]) - int(line[1]) > 0 and int(line[2]) - int(line[0])]
    line_tmp1 = []
    line_tmp2 = []
    for line in lines_tmp:
        if line[3] - line[1] < 5:  # 横向
            line_tmp1.append([line[0]-10.0, line[1], line[2]+10.0, line[3]])
        elif line[3] - line[1] >= 5:  # 纵向
            line_tmp2.append([line[0], line[1]-10.0, line[2], line[3]+10.0])

    points_list = []
    for i, line1 in enumerate(line_tmp1):
        for j, line2 in enumerate(line_tmp2):
            if i == j:
                continue
            line1 = list(map(int, line1))
            line2 = list(map(int, line2))
            l1 = [(line1[0], line1[2]), (line1[1], line1[3])]
            l2 = [(line2[0], line2[2]), (line2[1], line2[3])]
            t_cross = LineString(l1).intersects(LineString(l2))  # T, L
            shi_cross = LineString(l1).crosses(LineString(l2))   # 十

            if shi_cross or t_cross:
                points_list.append(line1)
                cv2.line(image, (line1[0], line1[1]), (line1[2], line1[3]), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.line(image, (line2[0], line2[1]), (line2[2], line2[3]), (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imwrite(r'E:\December\chinese\unblank\save2.jpg', image)
    print(points_list)
    return points_list


def clean_repeat_lines(lines):
    # 清除重复的线条
    for lineindex, line in enumerate(lines):
        if line[0] < 0:
            lines[lineindex][0] = -line[0]  # rho
            lines[lineindex][1] = line[1]-np.pi  # theta
    newlines = lines
    l_lines = len(lines)
    i = 0
    while i < l_lines:
        flag = 0
        j = i+1
        while j < l_lines:
            flag = 0
            if (abs(lines[i][1]-lines[j][1])<0.1):
                flag = 1
                lines.pop(j)
                l_lines -= 1
            j += 1
        newlines = lines
        i += 1
    return newlines


def FLD_detect(image):
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
    start_time = time.time()
    new_lines = clean_repeat_lines_(new_lines)
    end_time = time.time()
    cost_time = end_time - start_time
    print('cost_time:', cost_time)

    points = get_intersection_points(new_lines, image)
    print(points)

    # for dline in new_lines:
    #     x0 = int(round(dline[0]))
    #     y0 = int(round(dline[1]))
    #     x1 = int(round(dline[2]))
    #     y1 = int(round(dline[3]))
    #     cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imshow("FLD", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(r'E:\December\chinese\unblank\save1.jpg', image)


if __name__ == '__main__':
    img_path = r'E:\December\chinese\unblank\1.jpg'
    image = cv2.imread(img_path)
    FLD_detect(image)