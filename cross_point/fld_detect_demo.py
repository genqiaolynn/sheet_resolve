# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/4/16 19:07'


import cv2, time
import numpy as np


class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


def cross(p1, p2, p3):
    x1 = p2.x-p1.x
    y1 = p2.y-p1.y
    x2 = p3.x-p1.x
    y2 = p3.y-p1.y
    return x1*y2-x2*y1


def IsIntersec(p1, p2, p3, p4): #判断两线段是否相交
    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1.x, p2.x) >= min(p3.x, p4.x)    #矩形1最右端大于矩形2最左端
    and max(p3.x, p4.x) >= min(p1.x, p2.x)   #矩形2最右端大于矩形最左端
    and max(p1.y, p2.y) >= min(p3.y, p4.y)   #矩形1最高端大于矩形最低端
    and max(p3.y, p4.y) >= min(p1.y, p2.y)): #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验
        if(cross(p1, p2, p3) * cross(p1, p2, p4) <= 0 and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0):
            D = 1
        else:
            D = 0
    else:
        D = 0
    return D


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
    return newlines


def fld_detect(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(img_gray)
    lines = [line[0] for line in dlines.tolist()]

    # lines_arr = np.array(lines)
    # width_ = lines_arr[:, 2] - lines_arr[:, 0]
    # height_ = lines_arr[:, 3] - lines_arr[:, 1]
    # lines_index1 = np.where(width_ > 50)[0]
    # lines_index2 = np.where(height_ > 50)[0]
    # lines_index = np.hstack([lines_index1, lines_index2])
    # new_lines = [lines[ele] for ele in lines_index]
    # start_time = time.time()
    # new_lines = clean_repeat_lines(new_lines)
    # end_time = time.time()
    # cost_time = end_time - start_time
    # print('cost_time:', cost_time)

    # # 交叉点和开口方向
    # lines_tmp1 = []
    # lines_tmp2 = []
    # lines_tmp3 = []
    # for line in new_lines:
    #     if int(line[3]) - int(line[1]) < 5:   # 横排
    #         lines_tmp1.append([line[0] - 10.0, line[1], line[2] + 10.0, line[1]])
    #     elif int(line[2]) - int(line[0]) < 5:
    #         lines_tmp2.append([line[0], line[1] - 10.0, line[0], line[3] + 10.0])
    #     else:
    #         lines_tmp3.append(line)
    # if len(lines_tmp3) != 0:
    #     for ele in lines_tmp3:
    #         ww = ele[2] - ele[0]
    #         hh = ele[3] - ele[1]
    #         if ww > hh:
    #             lines_tmp1.append(ele)
    #         else:
    #             lines_tmp2.append(ele)
    # # 判断有交点
    # for line1 in lines_tmp1:
    #     for line2 in lines_tmp2:
    #         line1 = list(map(int, line1))
    #         line2 = list(map(int, line2))
    #         # l1 = [(line1[0], line1[2]), (line1[1], line1[3])]
    #         # l2 = [(line2[0], line2[2]), (line2[1], line2[3])]
    #
    #         if IsIntersec((line1[0], line1[1]), (line1[2], line1[3]), (line2[0], line2[1]), (line2[2], line2[3])) == 1:
    #             print('xiang')

    for dline in lines:
        x0 = int(round(dline[0]))
        y0 = int(round(dline[1]))
        x1 = int(round(dline[2]))
        y1 = int(round(dline[3]))
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(r'E:\December\chinese\unblank\save_.jpg', image)


if __name__ == '__main__':
    img_path = r'E:\December\chinese\unblank\1.jpg'
    image = cv2.imread(img_path)
    fld_detect(image)