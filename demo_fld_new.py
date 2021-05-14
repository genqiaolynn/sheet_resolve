# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/5/11 18:15'

from sklearn.cluster import DBSCAN
import cv2, time, heapq, os, copy, shutil, itertools
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon, LineString
import xml.etree.ElementTree as ET
from utils import *
from correct import *
from itertools import combinations


def clean_repeat_lines(lines):
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


def clean_repeat_lines_for(lines):
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


def lines_overlaps(line1, line2):
    overlaps = False
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # line1
    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - k1 * x1 * 1.0

    # line2
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - k2 * x3 * 1.0

    if not k1 is None and not k2 is None:
        if k1 == k2 and abs(b1 - b2) <= 0.1:
            overlaps = True
        else:
            overlaps = False
    return overlaps


def decide_k_and_b(line):
    x1, y1, x2, y2 = line
    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - k1 * x1 * 1.0
    return k1, b1


def decide_k_and_b_(line):
    x1, y1, x2, y2 = line
    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - k1 * x1 * 1.0
    return k1, b1


def decide_k_and_b_matrix(matrix):
    k1 = (matrix[:, 3] - matrix[:, 1]) * 1.0 / (matrix[:, 2] - matrix[:, 0])
    b1 = matrix[:, 1] * 1.0 - k1 * matrix[:, 0] * 1.0
    return k1, b1


# 基本全局阈值计算
def basic_global_threshold(hist):
    #i,t,t1,t2,k1,k2
    t=0;u=0;
    for idx,val in enumerate(hist):
        t+=val
        u+=idx*val
    k2 = int(u/t)
    k1= 0
    while(k1!=k2):
        k1=k2
        t1=0;u1=0;
        for idx,val in enumerate(hist):
            if(idx>k1):
                break
            t1+=val
            u1+=idx*val
        t2=t-t1
        u2=u-u1
        if t1: u1=u1/t1
        else: u1=0
        if t2: u2=u2/t2
        else: u2=0
        k2=int((u1+u2)/2)
    return k1


def custom_basic_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #把输入图像灰度化
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_arr = np.asarray(hist).astype(np.int32).flatten()
    mean = basic_global_threshold(hist_arr)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY_INV)
    return binary,mean


def convert_pil_to_jpeg(raw_img):
    if raw_img.mode == 'L':   # L是二值图像
        channels = raw_img.split()
        img = Image.merge("RGB", (channels[0], channels[0], channels[0]))
    elif raw_img.mode == 'RGB':
        img = raw_img
    elif raw_img.mode == 'RGBA':
        img = Image.new("RGB", raw_img.size, (255, 255, 255))
        img.paste(raw_img, mask=raw_img.split()[3])  # 3 is the alpha channel
    elif raw_img.mode == 'P':
        img = raw_img.convert('RGB')
    else:
        img = raw_img
    open_cv_image = np.array(img)
    return img, open_cv_image


def judge_cross_point_and_direction(points_list, image, name, pixel_thresh=6):
    height, width = image.shape[:2]
    imgBin, threshold = custom_basic_threshold(image)
    print('threshold:', threshold)
    if threshold < 127:
        thresholdV = 127
    else:
        thresholdV = threshold
    gray_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_, thresholdV, 1, cv2.THRESH_BINARY_INV)
    # ret, binary = cv2.threshold(gray_, 230, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_integral = cv2.integral(binary)   # 积分图
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'img_integral.jpg', img_integral)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'binary.jpg', binary)
    # 以上得到积分图  https://blog.csdn.net/weixin_41695564/article/details/80056430
    # 定义超参

    line_length = 20  # 交叉点边线长度
    ch = 2  # 误差宽度 一半
    adjust = 15  # 交叉点微调区域
    preSplit = 0.8
    totalPixes = line_length
    cross_point_box = []   # 扩增的图

    for point in points_list:
        left = point[0] - adjust if point[0] - adjust > 0 else 0
        top = point[1] - adjust if point[1] - adjust > 0 else 0

        right = point[0] + adjust if point[0] + adjust < width else width - 1
        bottom = point[1] + adjust if point[1] + adjust < height else height - 1
        cross_point_box.append([left, top, right, bottom])

    cross_point_list = []
    # 遍历这些每个扩增的box
    for box in cross_point_box:
        count = 0
        for y in range(box[1], box[3]):
            for x in range(box[0], box[2]):
                current_pixel = binary[y][x]
                if current_pixel == 0:
                    continue
                count = count + 1
                # image = cv2.circle(image, (x, y), 20, (0, 0, 244), 1, 8, 0)
                # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'image.jpg', image)
                # print(current_pixel)
                # 上下左右判断
                flag = [0, 0, 0, 0]
                trueCount = 0
                fri = [0.0, 0.0, 0.0, 0.0]
                # up
                left = x - ch  # 这里应该有安全判断 暂时不做
                top = y - line_length
                right = x + ch
                bottom = y
                oneCount = 0
                for n in range(top, bottom + 1):
                    fontCount = img_integral[n + 1][right] + img_integral[n][left] - img_integral[n][right] - img_integral[n + 1][left]  # 积分图
                    if (fontCount != 0):
                        oneCount += 1
                fri[0] = oneCount / totalPixes
                if (fri[0] > preSplit):
                    flag[0] = 1
                    trueCount += 1

                # down
                left = x - ch  # 这里应该有安全判断 暂时不做
                top = y
                right = x + ch
                bottom = y + line_length
                oneCount = 0
                for n in range(top, bottom):
                    fontCount = img_integral[n + 1][right] + img_integral[n][left] - img_integral[n][right] - img_integral[n + 1][left]
                    if (fontCount != 0):
                        oneCount += 1
                fri[1] = oneCount / totalPixes
                if (fri[1] > preSplit):
                    flag[1] = 1
                    trueCount += 1

                # left
                left = x - line_length  # 这里应该有安全判断 暂时不做
                top = y - ch
                right = x
                bottom = y + ch
                oneCount = 0
                for n in range(left, right + 1):
                    fontCount = img_integral[bottom][n + 1] + img_integral[top][n] - img_integral[top][n + 1] - img_integral[bottom][n]
                    if (fontCount != 0):
                        oneCount += 1
                fri[2] = oneCount / totalPixes
                if (fri[2] > preSplit):
                    flag[2] = 1
                    trueCount += 1

                # right
                left = x  # 这里应该有安全判断 暂时不做
                top = y - ch
                right = x + line_length if x + line_length < width else width
                bottom = y + ch
                oneCount = 0
                for n in range(left-1, right):
                    fontCount = img_integral[bottom][n + 1] + img_integral[top][n] - img_integral[top][n + 1] - img_integral[bottom][n]
                    if (fontCount != 0):
                        oneCount += 1
                fri[3] = oneCount / totalPixes
                if (fri[3] > preSplit):
                    flag[3] = 1
                    trueCount += 1

                if (trueCount < 2):
                    continue
                elif (trueCount == 2):
                    if ((flag[0] and flag[1]) or (flag[2] and flag[3])):
                        continue
                if trueCount >= 2:
                    cross_point_dict = {}
                    cross_point_dict['point'] = (x, y)
                    cross_point_dict['direction'] = flag
                    cross_point_dict['confidence'] = fri[0]+fri[1]+fri[2]+fri[3]
                    cross_point_list.append(cross_point_dict)

    return cross_point_list


def center_point(points):
    if len(points) == 0:
        return None
    dims = len(points[0])
    size = len(points)
    return tuple([sum([p[d] for p in points]) / size for d in range(dims)])


def clustering_points1(points, max_gap, center_trans=lambda x: int(round(x))):
    navigation = DBSCAN(eps=max_gap, min_samples=2).fit(points)
    core_samples_mask = np.zeros_like(navigation.labels_, dtype=bool)
    core_samples_mask[navigation.core_sample_indices_] = True
    labels = navigation.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    index_list = [i for i in range(n_clusters_)]
    clusters = {}
    for i in index_list:
        index_ = np.where(labels == i)[0]
        clusters[points[index_[0]]] = [points[ele] for ele in list(index_)]

    for g, s in clusters.items():
        c = center_point(s)
        del clusters[g]
        clusters[tuple([center_trans(i) for i in list(c)])] = s
    return clusters


def flatten(coll):
    flat = []
    for e in coll:
        flat.extend(e)
    return flat


def sort1(coll, key=lambda x: x, reverse=False):
    coll.sort(key=key, reverse=reverse)
    return coll


# 自适应阈值出来了很多噪点,感觉更适合自然场景多些
def prepare_gray_(img_color):
    img_gray = cv2.bitwise_not(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY))
    return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)


def prepare_gray(img_color):
    gray_img = cv2.bitwise_not(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY))   # 像素反转
    # cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    # return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2)
    return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # _ret, thresh_img = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # return thresh_im


def clustering_points(points, max_gap,
                      norm=np.linalg.norm,
                      center_trans=lambda x: int(round(x))):
    cluster = {}
    for point in points:
        if len(cluster) == 0:
            cluster[point] = [point]
        else:
            temp = [(i, min([norm(np.array(point) - np.array(p)) for p in group])) for i, group in cluster.items()]
            temp.sort(key=lambda d: d[1])
            i, dist = temp[0]
            if dist <= max_gap:
                cluster[i].append(point)
            else:
                cluster[point] = [point]
    for g, s in cluster.items():
        c = center_point(s)
        del cluster[g]
        cluster[tuple([center_trans(i) for i in list(c)])] = s
    return cluster


def dict_get(d, key, default):
    if key not in d:
        res = default()
        d[key] = res
        return res
    else:
        return d[key]


def dist_point_line(point, line,
                    is_line_segment=True,
                    norm=np.linalg.norm):
    p = np.array(point)
    a, b = [np.array(end) for end in line]
    ab = b - a
    ap = p - a
    d = norm(ab)
    r = 0 if d == 0 else ab.dot(ap) / (d ** 2)
    ac = r * ab
    if is_line_segment:
        if r <= 0:
            return norm(ap)
        elif r >= 1:
            return norm(p - b)
        else:
            return norm(ap - ac)
    else:
        return norm(ap - ac)


def groupby(coll, key):
    res = {}
    for e in coll:
        k = key(e)
        dict_get(res, k, lambda: []).append(e)
    return res


def group_reverse_map(group_res,
                      value=lambda v: v, key=lambda k: k):
    """
    Convert
    {
        k1:[u1,u2,...],
        k2:[v1,v2,...],
        ...
    }
    To
    {
        u1:k1,
        u2:k1,
        ...,
        v1:k2,
        v2:k2,
        ...
    }
    """
    return dict([(value(v), key(g)) for g, l in group_res.items() for v in l])


def imshow(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def maxindex(coll):
    """
    返回集合中最大值的下标
    """
    return None if len(coll) == 0 else coll.index(max(coll))


def minindex(coll):
    """
    返回集合中最小值的下标
    """
    return None if len(coll) == 0 else coll.index(min(coll))


def polygon_to_box(polygon):
    print(polygon)
    return (polygon[0], polygon[3], polygon[4], polygon[7])


def sort(coll, key=lambda x: x, reverse=False):
    coll.sort(key=key, reverse=reverse)
    return coll


# def outline_frame(img_gray, border_thickness, horizontal_scale=20.0, vertical_scale=20.0):
def outline_frame(img_gray, subject, border_thickness=4):
    # 语文这个阈值调小点25,因为作文好多线   其他学科暂时觉得越大越好,能找到足够多的点再去删
    if subject == '语文':
        horizontal_scale = 30
        vertical_scale = 30
    else:
        horizontal_scale = 50
        vertical_scale = 50
    (height, width) = img_gray.shape

    dilate_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (border_thickness, border_thickness))
    erode_structure = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # 勾勒横向直线
    horizontal = img_gray.copy()
    horizontal_size = int(width / horizontal_scale)
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, dilate_structure)
    horizontal = cv2.erode(horizontal, erode_structure)

    # 勾勒纵向直线
    vertical = img_gray.copy()
    vertical_size = int(height / vertical_scale)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, dilate_structure)
    vertical = cv2.erode(vertical, erode_structure)

    return horizontal, vertical


def merge_lines1(lines):
    lines = [sort([(line[0], line[1]), (line[2], line[3])]) for line in lines]
    ends = set(flatten(lines))
    ends_map = group_reverse_map(clustering_points(ends, 6))
    merged_set = set([tuple(sort([ends_map[line[0]], ends_map[line[1]]])) for line in lines])
    return [(line[0], line[1]) for line in merged_set]


def align_points(points):
    xs, ys = np.hsplit(np.array(points), 2)
    xg = clustering_points([(x[0],) for x in xs.tolist()], max_gap=6)
    xm = group_reverse_map(xg, lambda v: v[0], lambda k: k[0])
    yg = clustering_points([(y[0],) for y in ys.tolist()], max_gap=6)
    ym = group_reverse_map(yg, lambda v: v[0], lambda k: k[0])
    return [(xm[point[0]], ym[point[1]]) for point in points]


def filter_point(points_enlarge_tmp):
    # points_enlarge_tmp = points_list.copy()
    new_box_list = []
    while len(points_enlarge_tmp) > 0:
        xmin = points_enlarge_tmp[:, 0]
        ymin = points_enlarge_tmp[:, 1]
        xmax = points_enlarge_tmp[:, 2]
        ymax = points_enlarge_tmp[:, 3]
        area = (xmax - xmin) * (ymax - ymin)
        former = points_enlarge_tmp[0]
        index = [i for i in range(len(points_enlarge_tmp))]
        former = np.array(former)
        xxmin = np.maximum(former[0], xmin[1:])
        yymin = np.maximum(former[1], ymin[1:])
        xxmax = np.minimum(former[2], xmax[1:])
        yymax = np.minimum(former[3], ymax[1:])
        w = np.maximum(0.0, xxmax - xxmin)
        h = np.maximum(0.0, yymax - yymin)
        intersection = w * h
        ovp = intersection / (area[0] + area[1:] - intersection)
        indx = np.where(ovp >= 0.1)[0] + 1
        # print(indx)
        # print(len(points_enlarge_tmp))
        points_enlarge_t = [points_enlarge_tmp[ele] for ele in list(indx)]
        points_enlarge_t_array = np.array(points_enlarge_t)
        new_box = [np.min(points_enlarge_t_array[:, 0]), np.min(points_enlarge_t_array[:, 1]),
                   np.max(points_enlarge_t_array[:, 2]), np.max(points_enlarge_t_array[:, 3])]
        new_box_list.append(new_box)
        indx1 = list(indx)
        indx1.insert(0, 0)
        index_tmp = [ele for ele in index if ele not in indx1]
        points_enlarge_tmp = np.array([points_enlarge_tmp[ele] for ele in index_tmp])

        # new_box = [points_enlarge_tmp[ele] for ele in indx1]
        # for ele in [new_box]:
        #     cv2.rectangle(image, (int(ele[0]), int(ele[1])), (int(ele[2]), int(ele[3])), (0, 0, 255), 1)
        # write_single_img(image, os.path.join(save_path, str('point') + '_' + file))
    return new_box_list


def fld_demo4(image, subject, save_path, file, file_path):
    image_raw = image.copy()
    height, width = image.shape[:2]
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # fusion_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1)

    # img_gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # dist = cv2.distanceTransform(binary, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
    # write_single_img(image, os.path.join(save_path, str('dist') + '_' + file))
    # edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    # write_single_img(image, os.path.join(save_path, str('edges') + '_' + file))

    # TODO FLD直线检测是很耗时的,空白答题卡,三栏的那种需要0.16s一张,作答过的答题卡更耗时
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(binary)
    lines = [line[0] for line in dlines.tolist()]
    # e = time.time()
    # cost_time = e - s
    # print('cost_time:', cost_time)

    # TODO LSD直线检测  同样一张图fld和lsd检测的耗时情况:lsd更耗时,差不多要0.61s左右
    # lsd = cv2.createLineSegmentDetector(0, _scale=1)
    # dlines = lsd.detect(binary)
    # lines_ = [line for line in dlines[0]]
    # lines = [[int(round(ele[0][0])), int(round(ele[0][1])),
    # int(round(ele[0][2])), int(round(ele[0][3]))] for ele in lines_]
    # e = time.time()
    # cost_time = e - s
    # print('cost_time:', cost_time)

    lines_arr = np.array(lines)
    width_ = lines_arr[:, 2] - lines_arr[:, 0]
    height_ = lines_arr[:, 3] - lines_arr[:, 1]
    lines_index_h = np.where(width_ >= 75)[0]   # 一般认为没有那么短的横线
    lines_index_v = np.where(height_ >= 75)[0]  # 一般认为没有那么短的竖线
    lines_index = np.hstack([lines_index_h, lines_index_v])
    new_lines1 = [lines[ele] for ele in lines_index]

    # TODO 去掉重复的线 v1
    # frame_lines = merge_lines1(new_lines1)
    # new_lines = [[round(ele[0][0]), round(ele[0][1]), round(ele[1][0]), round(ele[1][1])] for ele in frame_lines]

    # TODO 去掉重复的线 v2
    new_lines = clean_repeat_lines(new_lines1)

    # for line in new_lines:
    #     cv2.line(image, (round(line[0]), round(line[1])), (round(line[2]), round(line[3])), (255, 255, 0), 2)
    # write_single_img(image, os.path.join(save_path, str('line') + '_' + file))

    # TODO  直线不去重了
    # new_lines = new_lines1

    # 拿水平线和垂直线
    horizontal_lines = []
    vertical_lines = []
    for ele in new_lines:
        if abs(ele[3] - ele[1]) > abs(ele[2] - ele[0]):
            vertical_lines.append(ele)
        else:
            horizontal_lines.append(ele)
    points_list = []
    for ele1 in horizontal_lines:
        for ele2 in vertical_lines:
            hor_line = LineString([(ele1[0], ele1[1]), (ele1[2], ele1[3])])
            ver_line = LineString([(ele2[0], ele2[1]), (ele2[2], ele2[3])])

            hor_line_extend = LineString([(1, ele1[1]), (width - 1, ele1[3])])
            ver_line_extend = LineString([(ele2[0], 1), (ele2[2], height - 1)])

            # cond1 = hor_line.intersects(ver_line)
            # cond2 = hor_line.crosses(ver_line)

            cond1 = hor_line_extend.intersects(ver_line_extend)
            cond2 = hor_line_extend.crosses(ver_line_extend)
            if True in [cond1, cond2]:
                (xp, yp) = hor_line_extend.intersection(ver_line_extend).bounds[:2]
                points_list.append((round(xp), round(yp)))
    # e = time.time()
    # cost_time = e - s
    # print('cost_time:', cost_time)
    # for ele in points_list:
    #     cv2.rectangle(image_raw, (int(ele[0]), int(ele[1])), (int(ele[0] + 1), int(ele[1]) + 1), (0, 0, 255), 1)
    # write_single_img(image_raw, os.path.join(save_path, str('point') + '_' + file))

    # e = time.time()
    # cost_time = e - s
    # print('cost_time:', cost_time)

    # points_list = clustering_points1(points_list, 6)
    # points_list_cluster1 = align_points(list(points_list.keys()))
    # for ele in points_list_cluster1:
    #     # cv2.rectangle(image_raw, (int(ele[0]), int(ele[1])), (int(ele[0] + 1), int(ele[1]) + 1), (0, 0, 255), 3)
    #     cv2.circle(image_raw, ele, 3, (0, 0, 233), 2, 8, 0)
    # write_single_img(image_raw, os.path.join(save_path, str('point_align') + '_' + file))

    # 每个点往四周扩,得到一个区域,点取的并集
    points_enlarge_list = [[ele[0]-20, ele[1]-20, ele[0]+20, ele[1]+20] for ele in points_list]
    points_enlarge_tmp = np.array(points_enlarge_list)
    # for ele in points_enlarge_list:
    #     cv2.rectangle(image, (int(ele[0]), int(ele[1])), (int(ele[2]), int(ele[3])), (0, 0, 255), 1)
    # write_single_img(image, os.path.join(save_path, str('enlarge') + '_' + file))

    e = time.time()
    cost_time = e - s
    print('cost_time:', cost_time)

    points_list = filter_point(points_enlarge_tmp)

    e = time.time()
    cost_time = e - s
    print('cost_time:', cost_time)
    print('找方向的区域有这么多个:', len(points_list))
    point_list_ = judge_cross_point_and_direction(points_list, image, file)
    points_list_all = [ele['point'] for ele in point_list_]

    e = time.time()
    cost_time = e - s
    print('cost_time:', cost_time)

    # for ele in points_list_all:
    #     cv2.rectangle(image_raw, (int(ele[0]), int(ele[1])), (int(ele[0] + 1), int(ele[1]) + 1), (0, 0, 255), 3)
    # write_single_img(image_raw, os.path.join(save_path, str('after_direction') + '_' + file))

    points_list_cluster = clustering_points1(points_list_all, 6)

    e = time.time()
    cost_time = e - s
    print('cost_time:', cost_time)



    # points_list_cluster_ = list(combinations(points_list_cluster, 2))
    # print('points_list_cluster_:', points_list_cluster_)
    # print(points_list_cluster)
    #
    # final_point_list = []
    # for raw_index, ele in enumerate(list(points_list_cluster.keys())):
    #     index_p = 0
    #     if ele in points_list_all:
    #         index_p = points_list_all.index(ele)  # 原始点
    #     else:
    #         point_x_ = [ele[0] + i for i in range(-10, 10)]
    #         point_y_ = [ele[1] + i for i in range(-10, 10)]
    #         point_xy = (list(itertools.product(point_x_, point_y_)))  # 聚类出来的点
    #
    #         for element in point_xy:
    #             if element in points_list_all:
    #                 index_p = points_list_all.index(element)
    #                 break
    #     # 聚类出来的点离原始的点同时平移多少个像素就保留原始点,判断聚类出来的点
    #     point_x_tmp = [ele[0] + i for i in range(-10, 10)]
    #     point_y_tmp = [ele[1] + i for i in range(-10, 10)]
    #     point_xy_tmp = (list(itertools.product(point_x_tmp, point_y_tmp)))  # 聚类出来的点
    #     # point_xy_tmp_array = np.array(point_xy_tmp)
    #
    #     # raw_point_list = [(ele[0] + i, ele[1] + i) for ele in cross_point_list for i in range(-6, 6)]
    #     # raw_point_array = np.array(raw_point_list)
    #     # max_dim = abs(point_xy_tmp_array.shape[0] - raw_point_array.shape[0])
    #     # if point_xy_tmp_array.shape[0] > raw_point_array.shape[0]:
    #     #     zero_mat = np.zeros((max_dim, 2), dtype=int)
    #     #     raw_point_array = np.vstack([raw_point_array, zero_mat])
    #     # else:
    #     #     zero_mat = np.zeros((max_dim, 2), dtype=int)
    #     #     point_xy_tmp_array = np.vstack([point_xy_tmp_array, zero_mat])
    #     # new_index1 = point_xy_tmp_array[np.logical_and.reduce(point_xy_tmp_array == raw_point_array, axis=1), :]
    #
    #     # new_index1 = np.arange(0, raw_point_array.shape[0])
    #     # new_index11 = new_index1[raw_point_array == point_xy_tmp_array]
    #
    #     # new_index1 = np.array(np.all((point_xy_tmp_array[:, None] == raw_point_array[None, :]), axis=-1).nonzero()).T.tolist()
    #     # point11 = [raw_point_array[ele[1]] for ele in list(new_index1)]
    #     for point_t in point_xy_tmp:
    #         if point_t in points_list:
    #             if abs(point_t[0] - ele[0]) >= 4 and abs(point_t[1] - ele[1]) >= 4:
    #                 new_point = tuple(point_t)  # 原始点
    #                 break
    #             else:
    #                 new_point = ele
    #                 break
    #     else:
    #         new_point = ele
    #     # direction --> four
    #     final_point_dict = {}
    #     # final_point_dict['point'] = new_point
    #     final_point_dict['point'] = ele      # TODO modify
    #     final_point_dict['direction'] = point_list_[index_p]['direction']
    #     final_point_list.append(final_point_dict)

    final_point_list = []
    for raw_index, ele in enumerate(list(points_list_cluster.keys())):
        index_p = 0
        if ele in points_list_all:
            index_p = points_list_all.index(ele)  # 原始点
        else:
            point_x_ = [ele[0] + i for i in range(-10, 10)]
            point_y_ = [ele[1] + i for i in range(-10, 10)]
            point_xy = (list(itertools.product(point_x_, point_y_)))  # 聚类出来的点

            for element in point_xy:
                if element in points_list_all:
                    index_p = points_list_all.index(element)
                    break
        # direction --> four
        final_point_dict = {}
        final_point_dict['point'] = ele
        final_point_dict['direction'] = point_list_[index_p]['direction']
        final_point_list.append(final_point_dict)

    for ele in final_point_list:
        # print(point,direction,cross.confidence)
        point = ele['point']
        direction = ele['direction']
        cv2.circle(image, point, 20, (0, 0, 233), 2, 8, 0)
        if (direction[0]):
            cv2.line(image, point, (point[0], point[1] - 20), (0, 0, 244), 2, 8, 0)
        if (direction[1]):
            cv2.line(image, point, (point[0], point[1] + 20), (0, 0, 244), 2, 8, 0)
        if (direction[2]):
            cv2.line(image, point, (point[0] - 20, point[1]), (0, 0, 244), 2, 8, 0)
        if (direction[3]):
            cv2.line(image, point, (point[0] + 20, point[1]), (0, 0, 244), 2, 8, 0)

    # for ele in points_list_cluster:
    #     cv2.circle(image, ele, 20, (0, 0, 233), 2, 8, 0)
    save_path1 = os.path.join(save_path, subject)
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    write_single_img(image, os.path.join(save_path1, str(1) + '_' + file))
    e = time.time()
    cost_time = e - s
    print('cost_time:', cost_time)


    # image1 = image.copy()
    # image2 = image.copy()
    # for line in horizontal_lines:
    #     cv2.line(image1, (round(line[0]), round(line[1])), (round(line[2]), round(line[3])), (255, 255, 0), 2)
    # for line in vertical_lines:
    #     cv2.line(image2, (round(line[0]), round(line[1])), (round(line[2]), round(line[3])), (255, 255, 0), 2)
    # save_path1 = os.path.join(save_path, subject)
    # if not os.path.exists(save_path1):
    #     os.makedirs(save_path1)
    # write_single_img(image1, os.path.join(save_path1, str(1) + '_' + file))
    # write_single_img(image2, os.path.join(save_path1, str(2) + '_' + file))


if __name__ == '__main__':
    img_path = r'E:\111_4_26_test_img\images'
    save_path = r'E:\111_4_26_test_img\save2'
    for root, dirs, files in os.walk(img_path):
        for file in files:
            file_path = os.path.join(root, file)
            subject = file_path.split('\\')[-2]
            img = Image.open(file_path)
            img_pil, image = convert_pil_to_jpeg(img)
            s = time.time()
            fld_demo4(image, subject, save_path, file, file_path)
