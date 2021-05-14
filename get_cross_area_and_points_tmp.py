# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/5/8 15:41'


import cv2, os, shutil, re, time, kobe
import itertools
import numpy as np
from utils import create_xml, crop_region_direct, read_single_img, write_single_img
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.cluster import DBSCAN


def remove_stamp(color_image, min_threshold=210, target_channel=2):
    cs = cv2.split(color_image)
    _, stamp = cv2.threshold(cs[target_channel], min_threshold, 255, cv2.THRESH_BINARY)
    nostamp = cv2.merge([cv2.bitwise_or(c, stamp) for c in cs])
    # _, nostamp = cv2.threshold(cs[target_channel], min_threshold, 255, cv2.THRESH_BINARY)
    return nostamp


def center_point(points):
    if len(points) == 0:
        return None
    dims = len(points[0])
    size = len(points)
    return tuple([sum([p[d] for p in points]) / size for d in range(dims)])


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


def clustering_points1(points, max_gap, center_trans=lambda x: int(round(x))):
    navigation = DBSCAN(eps=max_gap, min_samples=3).fit(points)
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


def flatten(coll):
    flat = []
    for e in coll:
        flat.extend(e)
    return flat


def groupby(coll, key):
    res = {}
    for e in coll:
        k = key(e)
        dict_get(res, k, lambda: []).append(e)
    return res


def group_reverse_map(group_res, value=lambda v: v, key=lambda k: k):
    return dict([(value(v), key(g)) for g, l in group_res.items() for v in l])


def imshow(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def maxindex(coll):
     return None if len(coll) == 0 else coll.index(max(coll))


def minindex(coll):
    return None if len(coll) == 0 else coll.index(min(coll))


def polygon_to_box(polygon):
    print(polygon)
    return (polygon[0], polygon[3], polygon[4], polygon[7])


def sort(coll, key=lambda x: x, reverse=False):
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
    # return thresh_img


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


def align_points(points):
    xs, ys = np.hsplit(np.array(points), 2)
    xg = clustering_points([(x[0],) for x in xs.tolist()], max_gap=5)
    xm = group_reverse_map(xg, lambda v: v[0], lambda k: k[0])
    yg = clustering_points([(y[0],) for y in ys.tolist()], max_gap=5)
    ym = group_reverse_map(yg, lambda v: v[0], lambda k: k[0])
    return [(xm[point[0]], ym[point[1]]) for point in points]


def cross_points(frame_h, frame_v):
    # TODO 交叉点 与 角点 取并集
    cross_point = cv2.bitwise_and(frame_h, frame_v)
    cross_ys, cross_xs = np.where(cross_point > 0)
    return align_points(list(clustering_points1(list(zip(cross_xs, cross_ys)), 6).keys()))


def get_std_row(img_mtx):
    # img_mtx = np.asarray(rgb2binary(image))
    x_sum_array = img_mtx.sum(axis=0)
    std_row = np.std(x_sum_array)  # 计算标准差
    mean = round(np.mean(img_mtx) / 255, 6)
    return std_row, mean


def get_std_column(img_mtx):
    # img_mtx = np.asarray(rgb2binary(image))
    y_sum_array = img_mtx.sum(axis=1)
    std_column = np.std(y_sum_array)  # 计算标准差
    mean = round(np.mean(img_mtx) / 255, 6)
    return std_column, mean


def judge_cross_point_and_direction_raw(points_list, binary, name, thresh=9, pixel_thresh=0):
    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite(r'E:\111_4_26_test_img\aa\222\thresh_img.jpg', thresh_img)

    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'ada_box.jpg', binary)
    # 判断这个点是不是交叉点
    points_list_tmp = []
    for ele in points_list:
        point_left = ele[0] - thresh
        point_right = ele[0] + thresh
        point_top = ele[1] - thresh
        point_bottom = ele[1] + thresh

        DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = False, False, False, False
        # left region
        hei_pixel_left = binary[int(ele[1]) - 2:int(ele[1]) + 2, int(point_left): int(ele[0])]
        std1, mean1 = get_std_column(hei_pixel_left)

        # cv2.rectangle(image, (int(point_left), int(ele[1]) - 2), (int(ele[0]), int(ele[1]) + 2), (255, 0, 255), 1)
        # cv2.putText(image, str(std1), (int(ele[0]), int(ele[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        # cv2.imwrite(r'E:\111_4_26_test_img\aa\save2\\' + name + '_std_box.jpg', image)

        if std1 > pixel_thresh:
            DIR_LEFT = True
        # right region
        hei_pixel_right = binary[int(ele[1]) - 2:int(ele[1]) + 2, int(ele[0]): int(point_right)]
        std2, mean2 = get_std_column(hei_pixel_right)
        if std2 > pixel_thresh:
            DIR_RIGHT = True

        # top region
        hei_pixel_top = binary[int(point_top): int(ele[1]), int(ele[0]) - 2:int(ele[0]) + 2]
        std3, mean3 = get_std_row(hei_pixel_top)
        if std3 > pixel_thresh:
            DIR_UP = True

        # bottom region
        hei_pixel_bottom = binary[int(ele[1]): int(point_bottom), int(ele[0]) - 2:int(ele[0]) + 2]
        std4, mean4 = get_std_row(hei_pixel_bottom)
        if std4 > pixel_thresh:
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
        std_list = [std1, std2, std3, std4]
        mean_list = [mean1, mean2, mean3, mean4]
        rule_list_tmp = [CROSS_TL, CROSS_TR, CROSS_TLR, CROSS_BL, CROSS_BR, CROSS_BLR, CROSS_TBL, CROSS_TBR, CROSS_TBLR]
        index_true = np.where(np.array(rule_list_tmp) == True)[0]
        if len(index_true) > 2:
            max_index = max(index_true)
            direction = direction_dict[max_index]
        elif len(index_true) == 0:
            continue
        elif len(index_true) == 1:
            direction = direction_dict[index_true[0]]
        else:
            continue
        point_direction_dict = {}
        point_direction_dict['point'] = ele
        point_direction_dict['direction'] = direction
        point_direction_dict['std'] = std_list
        point_direction_dict['mean'] = mean_list
        points_list_tmp.append(point_direction_dict)

        cv2.rectangle(image, (int(point_left), int(ele[1]) - 2), (int(ele[0]), int(ele[1]) + 2), (0, 0, 255), 1)
        cv2.rectangle(image, (int(ele[0]), int(ele[1]) - 2), (int(point_right), int(ele[1]) + 2), (0, 0, 255), 1)
        cv2.rectangle(image, (int(ele[0]) - 2, int(point_top)), (int(ele[0]) + 2, int(ele[1])), (0, 0, 255), 1)
        cv2.rectangle(image, (int(ele[0]) - 2, int(ele[1])), (int(ele[0]) + 2, int(point_bottom)), (0, 0, 255), 1)
        cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + '_box.jpg', image)
    return points_list_tmp


# 遍历像素
def ergodic_pixel(current_point_, binary, file, pixel_thresh=6):
    blank_pixel = []
    new_point_list = []
    std_list = []
    # 当前值是白色像素,直接判断附近的值,要是不是白色则遍历
    # index_list1 = [0, 1, -1, 2, -2, 3, -3, 4, -4]  # 先正的再负的  一般走到遍历像素的阶段,本身是判断过不行的,所以这边不加0
    # index_list1 = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]  # 先正的再负的  一般走到遍历像素的阶段,本身是判断过不行的,所以这边不加0

    index_list1 = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]  # 先正的再负的  一般走到遍历像素的阶段,本身是判断过不行的,所以这边不加0
    point_list_temp = []
    std_list_all = []
    early_stop_Flag = False
    for m in index_list1:
        if len(blank_pixel) > 0 or early_stop_Flag:
            break
        for n in range(-pixel_thresh // 2, pixel_thresh // 2):
            if len(blank_pixel) > 0 or early_stop_Flag:
                break
            point_x = current_point_[0]
            point_y = current_point_[1]
            current_point = (point_x + m, point_y + n)

            current_pixel = binary[point_y + n][point_x + m]
            # 避免判断重复的点
            if current_point in point_list_temp:
                continue
            point_list_temp.append(current_point)
            # 标准差5个里有三个都是四个方向的值,说明这个点可能是文字部分,直接跳过循环
            if current_pixel == 255:
                cond1, cond2, cond3, blank_pixel1, new_point_list1, std_list1 = pixel_near_point(current_point, binary, file)
                std_list_all.append(cond2)
                # 这个点可能在文字区域,遇到这样的点直接舍弃,不继续往下判断
                zz = std_list_all.count([0, 1, 2, 3])
                if len(std_list_all) == 5 and zz >= 3:
                    early_stop_Flag = True
                index_ = list(np.where(cond3 == True)[0])
                # [0, 2] and [1, 3] --> 一条直线上
                if cond1 == cond2 and len(index_) >= 2 and index_ != [0, 2] and index_ != [1, 3]:
                    blank_pixel = blank_pixel1
                    new_point_list = new_point_list1
                    std_list = std_list1
                else:
                    current_point_ = current_point_
    return blank_pixel, new_point_list, std_list


def pixel_near_point(current_point, binary, file, image):
    # 拿到当前点是白色像素,判断这个点四周的像素,判断方向
    line_width = 0
    pixel_thresh_l = 12   # 长
    pixel_thresh_s = 6    # 短
    # 形成的矩阵是12*6
    left_box = [int(current_point[0] - pixel_thresh_l), int(current_point[1]) - pixel_thresh_s // 2,
                int(current_point[0] - line_width), int(current_point[1]) + pixel_thresh_s // 2]
    # right box
    right_box = [int(current_point[0]) + line_width, int(current_point[1]) - pixel_thresh_s // 2,
                 int(current_point[0] + pixel_thresh_l), int(current_point[1]) + pixel_thresh_s // 2]
    # top box
    top_box = [int(current_point[0]) - pixel_thresh_s // 2, int(current_point[1] - pixel_thresh_l),
               int(current_point[0]) + pixel_thresh_s // 2, int(current_point[1] - line_width)]

    # bottom box
    bottom_box = [int(current_point[0]) - pixel_thresh_s // 2, int(current_point[1] + line_width),
                  int(current_point[0]) + pixel_thresh_s // 2, int(current_point[1] + pixel_thresh_l)]

    binary_left = binary[left_box[1]:left_box[3], left_box[0]:left_box[2]]
    left_blank_pixel = len(binary_left[binary_left == 255])
    y_sum_array_l = binary_left.sum(axis=1)
    std_column_left = np.std(y_sum_array_l)
    print(left_blank_pixel)

    binary_right = binary[right_box[1]:right_box[3], right_box[0]:right_box[2]]
    right_blank_pixel = len(binary_right[binary_right == 255])
    y_sum_array_r = binary_right.sum(axis=1)
    std_column_right = np.std(y_sum_array_r)
    print(right_blank_pixel)

    binary_top = binary[top_box[1]:top_box[3], top_box[0]:top_box[2]]
    top_blank_pixel = len(binary_top[binary_top == 255])
    y_sum_array_t = binary_top.sum(axis=0)
    std_column_top = np.std(y_sum_array_t)
    print(top_blank_pixel)

    binary_bottom = binary[bottom_box[1]:bottom_box[3], bottom_box[0]:bottom_box[2]]
    bottom_blank_pixel = len(binary_bottom[binary_bottom == 255])
    y_sum_array_b = binary_bottom.sum(axis=0)
    std_column_bottom = np.std(y_sum_array_b)
    print(bottom_blank_pixel)

    # 以上包含的区域全部都是白色像素的总数
    # TODO 以上区域内全部的白色像素总数
    box_tmp = [top_box[0] + 10, top_box[1] - 20, top_box[2] + 10, top_box[3] - 20]
    binary_top_tmp = image[box_tmp[1]:box_tmp[3], box_tmp[0]:box_tmp[2]]
    binary_top_tmp_pixel = len(binary_top_tmp[binary_top_tmp == 255])

    # cv2.rectangle(image, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(current_point) + '_box.jpg', image)
    #
    # cv2.rectangle(image, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(current_point) + '_box.jpg', image)
    #
    # cv2.rectangle(image, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(current_point) + '_box.jpg', image)
    #
    # cv2.rectangle(image, (bottom_box[0], bottom_box[1]), (bottom_box[2], bottom_box[3]), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(current_point) + '_box.jpg', image)

    bbox_list_tmp = [left_box, top_box, right_box, bottom_box, box_tmp]
    template = r'./exam_info/000000-template.xml'
    tree = ET.parse(template)
    for box_ in bbox_list_tmp:
        create_xml(str(box_) + '_', tree, int(box_[0]), int(box_[1]), int(box_[2]), int(box_[3]))

    tree.write(os.path.join(r'E:\111_4_26_test_img\save', file.replace('.jpg', '.xml')))
    cv2.imwrite(os.path.join(r'E:\111_4_26_test_img\save', file), image)

    pixel_tmp = [left_blank_pixel, top_blank_pixel, right_blank_pixel, bottom_blank_pixel]
    std_tmp = [std_column_left, std_column_top, std_column_right, std_column_bottom]
    cond1 = list(np.where((np.array(pixel_tmp) - int(np.mean((np.array(pixel_tmp))))) > 0)[0])
    cond2 = list(np.where(np.array(std_tmp) > 320)[0])
    cond3 = np.array(pixel_tmp) - int(np.mean(pixel_tmp)) > 0

    cond4 = list(np.where(np.array(pixel_tmp) == 0)[0])
    cond5 = list(np.where(np.array(std_tmp) == 0.0)[0])

    if cond2 == [1, 3] or cond2 == [0, 2]:   # 1, 3和2, 4代表在一条直线上
        cond2 = []

    blank_pixel = []
    new_point_list = []
    std_list = []

    blank_pixel.extend([left_blank_pixel, top_blank_pixel, right_blank_pixel, bottom_blank_pixel])
    new_point_list.extend(current_point)
    std_list.extend([std_column_left, std_column_top, std_column_right, std_column_bottom])
    return cond1, cond2, cond3, blank_pixel, new_point_list, std_list


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


# TODO 判断每个点附近的值 这种方法不好的原因是找不到那个最优的点
def judge_cross_point_and_direction_raw2(points_list, image, name, pixel_thresh=6):
    gray_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_, 230, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'erosion.jpg', binary)
    # 判断这个点是不是交叉点
    points_list_tmp = []
    for index, ele in enumerate(points_list):
        DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = False, False, False, False
        # TODO 向四个方向扩15个像素,形成一个30*30的矩形,判断矩形里的黑色像素,并判断方向

        # # left box
        # left_box = [int(ele[0] - pixel_thresh), int(ele[1]) - pixel_thresh, int(ele[0]), int(ele[1]) + pixel_thresh]
        # # right box
        # right_box = [int(ele[0]), int(ele[1]) - pixel_thresh, int(ele[0] + pixel_thresh), int(ele[1]) + pixel_thresh]
        # # top box
        # top_box = [int(ele[0]) - pixel_thresh, int(ele[1] - pixel_thresh), int(ele[0]) + pixel_thresh, int(ele[1])]
        # # bottom box
        # bottom_box = [int(ele[0]) - pixel_thresh, int(ele[1]), int(ele[0]) + pixel_thresh, int(ele[1] + pixel_thresh)]

        # 以点为中心形成的矩形
        # box_by_point = [int(ele[0]) - pixel_thresh, int(ele[1]) - pixel_thresh, int(ele[0]) + pixel_thresh, int(ele[1]) + pixel_thresh]
        # cv2.rectangle(image, (box_by_point[0], box_by_point[1]), (box_by_point[2], box_by_point[3]), (0, 0, 255), 1)

        # 遍历点左边的区域,判断黑色像素
        # 判断right和bottom,此时图像做了翻转,需要找白色像素
        # 先上下找,再左右找
        # i控制矩形, j是小矩形,以拿到的点为中心向四周找,直到找到交叉点为止,假设拿到的点都有问题,判断每一个拿到的点
        point_x = int(ele[0])
        point_y = int(ele[1])
        # if point_x == 2348 and point_y == 2782:
        current_pixel = binary[point_y][point_x]
        current_point = (point_x, point_y)

        # 拿到的点就是白色像素,只需判断附近的值
        if current_pixel == 255:
            # 拿到的点是白色像素
            cond1, cond2, cond3, blank_pixel1, new_point_list1, std_list1 = pixel_near_point(current_point, binary, file, image)
            index_1 = np.where(cond3 == True)[0]
            if cond1 == cond2 and len(index_1) >= 2:
                blank_pixel = blank_pixel1
                new_point_list = new_point_list1
            else:
                blank_pixel, new_point_list, std_list = ergodic_pixel(current_point, binary, file)
        else:
            # 拿到的点不是白色像素,需要遍历周围的点
            blank_pixel, new_point_list, std_list = ergodic_pixel(current_point, binary, file)
            # # 当前值是白色像素,直接判断附近的值,要是不是白色则遍历
            # for m in range(-pixel_thresh//2, pixel_thresh//2):
            #     if len(blank_pixel) > 0:
            #         break
            #     for n in range(-pixel_thresh//2, pixel_thresh//2):
            #         if len(blank_pixel) > 0:
            #             break
            #         current_point = (point_x + m, point_y + n)
            #         current_pixel = binary[point_y + n][point_x + m]
            #         if current_pixel == 255:
            #             # 拿到当前点是白色像素,判断这个点四周的像素,判断方向
            #             line_width = 0
            #             left_box = [int(current_point[0] - pixel_thresh), int(current_point[1]) - pixel_thresh//2,
            #                         int(current_point[0] - line_width), int(current_point[1]) + pixel_thresh//2]
            #             # right box
            #             right_box = [int(current_point[0]) + line_width, int(current_point[1]) - pixel_thresh//2,
            #                          int(current_point[0] + pixel_thresh), int(current_point[1]) + pixel_thresh//2]
            #             # top box
            #             top_box = [int(current_point[0]) - pixel_thresh//2, int(current_point[1] - pixel_thresh),
            #                        int(current_point[0]) + pixel_thresh//2, int(current_point[1] - line_width)]
            #
            #             # bottom box
            #             bottom_box = [int(current_point[0]) - pixel_thresh//2, int(current_point[1] + line_width),
            #                           int(current_point[0]) + pixel_thresh//2, int(current_point[1] + pixel_thresh)]
            #
            #             binary_left = binary[left_box[1]:left_box[3], left_box[0]:left_box[2]]
            #             left_blank_pixel = len(binary_left[binary_left == 255])
            #             y_sum_array_l = binary_left.sum(axis=1)
            #             std_column_left = np.std(y_sum_array_l)
            #             print(left_blank_pixel)
            #
            #             binary_right = binary[right_box[1]:right_box[3], right_box[0]:right_box[2]]
            #             right_blank_pixel = len(binary_right[binary_right == 255])
            #             y_sum_array_r = binary_right.sum(axis=1)
            #             std_column_right = np.std(y_sum_array_r)
            #             print(right_blank_pixel)
            #
            #             binary_top = binary[top_box[1]:top_box[3], top_box[0]:top_box[2]]
            #             top_blank_pixel = len(binary_top[binary_top == 255])
            #             y_sum_array_t = binary_top.sum(axis=0)
            #             std_column_top = np.std(y_sum_array_t)
            #             print(top_blank_pixel)
            #
            #             binary_bottom = binary[bottom_box[1]:bottom_box[3], bottom_box[0]:bottom_box[2]]
            #             bottom_blank_pixel = len(binary_bottom[binary_bottom == 255])
            #             y_sum_array_b = binary_bottom.sum(axis=0)
            #             std_column_bottom = np.std(y_sum_array_b)
            #             print(bottom_blank_pixel)
            #
            #             # cv2.rectangle(image, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 0, 255), 1)
            #             # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(current_point) + '_box.jpg', image)
            #             #
            #             # cv2.rectangle(image, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 0, 255), 1)
            #             # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(current_point) + '_box.jpg', image)
            #             #
            #             # cv2.rectangle(image, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (0, 0, 255), 1)
            #             # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(current_point) + '_box.jpg', image)
            #             #
            #             # cv2.rectangle(image, (bottom_box[0], bottom_box[1]), (bottom_box[2], bottom_box[3]), (0, 0, 255), 1)
            #             # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(current_point) + '_box.jpg', image)
            #
            #             bbox_list_tmp = [left_box, top_box, right_box, bottom_box]
            #             template = r'./exam_info/000000-template.xml'
            #             tree = ET.parse(template)
            #             for box_ in bbox_list_tmp:
            #                 create_xml(str(box_) + '_', tree, int(box_[0]), int(box_[1]), int(box_[2]), int(box_[3]))
            #
            #             tree.write(os.path.join(r'E:\111_4_26_test_img\save', file.replace('.jpg', '.xml')))
            #             cv2.imwrite(os.path.join(r'E:\111_4_26_test_img\save', file), image)
            #
            #             pixel_tmp = [left_blank_pixel, top_blank_pixel, right_blank_pixel, bottom_blank_pixel]
            #             std_tmp = [std_column_left, std_column_top, std_column_right, std_column_bottom]
            #             # TODO 5-6写的条件 基于像素8上
            #             # if list(np.where((np.array(pixel_tmp) - 10) < 0)[0]) != 0 and int(np.mean((np.array(pixel_tmp)))) < sorted(pixel_tmp)[1]\
            #             #         and int(np.mean(np.array(std_tmp)) > 5.0):
            #             # TODO 5-7写的条件
            #             cond1 = list(np.where((np.array(pixel_tmp) - int(np.mean((np.array(pixel_tmp)))) + 2) > 0)[0])
            #             cond2 = list(np.where(np.array(std_tmp) > 200)[0])
            #             cond3 = list(np.where(np.array(pixel_tmp) == 0)[0])
            #             cond4 = list(np.where(np.array(std_tmp) == 0.0)[0])
            #             if cond1 == cond2:
            #                 blank_pixel.append([left_blank_pixel, top_blank_pixel, right_blank_pixel, bottom_blank_pixel])
            #                 new_point_list.append(current_point)
            #                 mean1.append(int(np.mean((np.array(pixel_tmp)))))
            #                 std_list.append([std_column_left, std_column_top, std_column_right, std_column_bottom])

                    # if list(np.where((np.array(pixel_tmp) - int(np.mean((np.array(pixel_tmp))))) > 0)[0]) != 0 and 0.0 in std_tmp:
                    #         blank_pixel.append([left_blank_pixel, top_blank_pixel, right_blank_pixel, bottom_blank_pixel])
                    #         new_point_list.append(current_point)
                    #         mean1.append(int(np.mean((np.array(pixel_tmp)))))
                    #         std_list.append([std_column_left, std_column_top, std_column_right, std_column_bottom])


                    # cv2.rectangle(image, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 0, 255), 1)
                    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'left_box.jpg', image)
                    #
                    # cv2.rectangle(image, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 0, 255), 1)
                    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'right_box.jpg', image)
                    #
                    # cv2.rectangle(image, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (0, 0, 255), 1)
                    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'top_box.jpg', image)
                    #
                    # cv2.rectangle(image, (bottom_box[0], bottom_box[1]), (bottom_box[2], bottom_box[3]), (0, 0, 255), 1)
                    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'bottom_box.jpg', image)
        print(blank_pixel)
        if len(blank_pixel) == 0:
            continue
        true_ = np.array(blank_pixel) - int(np.mean(blank_pixel)) >= 0
        index_ = np.where(true_ == True)[0]
        if 0 in index_:
            DIR_LEFT = True
        if 1 in index_:
            DIR_UP = True
        if 2 in index_:
            DIR_RIGHT = True
        if 3 in index_:
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
        index_true = np.where(np.array(rule_list_tmp) == True)[0]
        if len(index_true) > 2:
            max_index = max(index_true)
            direction = direction_dict[max_index]
        elif len(index_true) == 0:
            continue
        elif len(index_true) == 1:
            direction = direction_dict[index_true[0]]
        else:
            continue
        point_and_direction_dict = {}
        point_and_direction_dict['point'] = new_point_list
        point_and_direction_dict['direction'] = direction
        points_list_tmp.append(point_and_direction_dict)

    return points_list_tmp


# 判断每个点临近的区域[2021.5.8新修改]
def judge_cross_point_and_direction(points_list, image, name, pixel_thresh=6):
    height, width = image.shape[:2]
    imgBin, threshold = custom_basic_threshold(image)
    if threshold <= 127:
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

    line_length = 15  # 交叉点边线长度
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


# TODO 5月10新增
def judge_cross_point_and_direction_(points_list, image, name, pixel_thresh=6):

    print(points_list)
    return points_list


def point_nearby_lines(point, lines, max_dist):
    return set([line for line in lines if dist_point_line(point, line) <= max_dist])


def cross_point_without_character_v1(cross_point, image):
    # 去掉不在矩形框上的点
    height, width = image.shape[:2]
    binary = prepare_gray(image)
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # 这里只找外接矩形
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt_id, cnt in enumerate(reversed(contours)):
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\aa.jpg', image)

    # 框出面积最大的几个就好
    bboxes_array = np.array(bboxes)
    bboxes_area = (bboxes_array[:, 2] - bboxes_array[:, 0]) * (bboxes_array[:, 3] - bboxes_array[:, 1])
    max_index = np.argsort(bboxes_area)[::-1][:8]
    bboxes_new = [bboxes_array[ele] for ele in list(max_index)]

    for ele in bboxes_new:
        cv2.rectangle(image, (ele[0], ele[1]), (ele[2], ele[3]), (0, 0, 255), 1)
        cv2.imwrite(r'E:\111_4_26_test_img\save\aa.jpg', image)
    print('ok')


def split_region(cross_points, point_lines):
    # 按行遍历交点
    cross_points.sort(key=lambda p: (p[1], p[0]))

    xm = groupby(cross_points, lambda p: p[0])
    for k, v in xm.items():
        xm[k] = [p[1] for p in v]

    ym = groupby(cross_points, lambda p: p[1])
    for k, v in ym.items():
        ym[k] = [p[0] for p in v]

    ps = set(cross_points)

    regions = []
    for a in cross_points:
        x, y = a
        b, c, d = None, None, None
        zp = None
        for v in [v for v in xm[x] if v > y]:
            d = (x, v)
            for u in [u for u in ym[y] if u > x]:
                b = (u, y)
                c = (u, v)
                if c in ps \
                        and len(point_lines[a].intersection(point_lines[b])) > 0 \
                        and len(point_lines[b].intersection(point_lines[c])) > 0 \
                        and len(point_lines[c].intersection(point_lines[d])) > 0 \
                        and len(point_lines[d].intersection(point_lines[a])) > 0:
                    zp = c
                    break
            if zp is not None:
                break
        if zp is not None:
            region = [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]]
            regions.append(region)

    return regions


def image_with_character_(image, filename):
    # 去掉不在矩形框上的点
    image_raw = image.copy()
    height, width = image.shape[:2]
    mser = cv2.MSER_create(_min_area=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(gray)

    # 第一遍筛选,保留长宽在[0, 10]像素之内的框
    bboxes1 = []
    for ele in boxes:
        x, y, w, h = ele
        if 0 < abs(w - h) <= 60:
            bboxes1.append([x, y, w+x, y+h])
    print(bboxes1)

    # 第二遍筛选,去掉这里框里黑色像数比较少得框,容易框到了边缘
    # binary = prepare_gray(image)  # 自适应阈值出来的图像有很多噪点,容易引起误判的情况
    gray_img = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    _ret, thresh_img = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bboxes2 = []
    for ele in bboxes1:
        small_region = thresh_img[ele[1]:ele[3], ele[0]:ele[2]]
        black_pixel = len(small_region[small_region == 0])
        blank_pixel = len(small_region[small_region == 255])
        std_ = np.std(small_region)
        if 0.5*blank_pixel < black_pixel < 20*blank_pixel and std_ > 100:
            bboxes2.append(ele)
    # 画图,看这时候出来的框是啥样的
    for box in bboxes2:
        small_region = thresh_img[box[1]:box[3], box[0]:box[2]]
        black_pixel = len(small_region[small_region == 0])
        blank_pixel = len(small_region[small_region == 255])
        std_row = np.std(small_region)  # 计算标准差
        cv2.rectangle(image_raw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(image_raw, str([black_pixel, blank_pixel, std_row]), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(1) + filename + '.jpg', image_raw)


    # 把上面拿出来的框对应的图像部分抹成白色像素
    # for bbox1 in bboxes2:
    #     image[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]] = 255
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(1) + filename + '.jpg', image)
    # xywh --> xyxy
    # bboxes1 = []
    # for ele in boxes:
    #     x, y, w, h = ele
    #     bboxes1.append([x, y, x+w, y+h])
    # bboxes1_array = np.array(bboxes1)
    # area = (bboxes1_array[:, 3] - bboxes1_array[:, 1]) * (bboxes1_array[:, 2] - bboxes1_array[:, 0])
    # index_ = np.argsort(area)[::-1]    # 从大到小
    # # boxes_ = [bboxes1[ele] for ele in list(index_[:50])]
    #
    # for box in bboxes1:
    #     # x, y, w, h = box
    #     if 0 < abs((box[2] - box[0]) - (box[3] - box[1])) <= 10:
    #         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    #         cv2.putText(image, str([box[0], box[1], box[2], box[3]]), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(1) + filename + '.jpg', image)
    #
    # binary = prepare_gray(image)
    # # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # 这里只找外接矩形
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # bboxes = []
    # for cnt_id, cnt in enumerate(reversed(contours)):
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     bboxes.append((x, y, x + w, y + h))
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\aa.jpg', image)

    # 框出面积最大的几个就好
    # bboxes_array = np.array(bboxes)
    # bboxes_area = (bboxes_array[:, 2] - bboxes_array[:, 0]) * (bboxes_array[:, 3] - bboxes_array[:, 1])
    # max_index = np.argsort(bboxes_area)[::-1][:8]
    # bboxes_new = [bboxes_array[ele] for ele in list(max_index)]
    #
    # for ele in bboxes_new:
    #     cv2.rectangle(image, (ele[0], ele[1]), (ele[2], ele[3]), (0, 0, 255), 1)
    #     cv2.imwrite(r'E:\111_4_26_test_img\save\aa.jpg', image)
    # print('ok')
    return image


def image_with_character_v1(image, filename):
    # TODO MSER
    # 去掉不在矩形框上的点
    image_raw = image.copy()
    image1 = image.copy()
    # mser = cv2.MSER_create(_min_area=300)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # regions, boxes = mser.detectRegions(gray)
    #
    # # 第一遍筛选,保留长宽在[0, 10]像素之内的框
    # bboxes1 = []
    # for ele in boxes:
    #     x, y, w, h = ele
    #     bboxes1.append([x, y, w+x, y+h])
    # print(bboxes1)
    # bboxes1_array = np.array(bboxes1)
    # area = (bboxes1_array[:, 2] - bboxes1_array[:, 0]) * (bboxes1_array[:, 3] - bboxes1_array[:, 1])
    # index_ = np.argsort(area)[::-1][:50]
    # bboxes2 = [bboxes1[ele] for ele in list(index_)]
    # for box in bboxes2:
    #     cv2.rectangle(image_raw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    #     cv2.putText(image_raw, str([box[0], box[1], box[2], box[3]]), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str('mser') + filename + '.jpg', image_raw)



    # # 第二遍筛选,去掉这里框里黑色像数比较少得框,容易框到了边缘
    # # binary = prepare_gray(image)  # 自适应阈值出来的图像有很多噪点,容易引起误判的情况
    # # gray_img = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    # gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # _ret, thresh_img = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # bboxes2 = []
    # for ele in bboxes1:
    #     small_region = thresh_img[ele[1]:ele[3], ele[0]:ele[2]]
    #     black_pixel = len(small_region[small_region == 0])
    #     blank_pixel = len(small_region[small_region == 255])
    #     std_ = np.std(small_region)
    #     if 0.5*blank_pixel < black_pixel < 20*blank_pixel and std_ > 100:
    #         bboxes2.append(ele)
    # # 画图,看这时候出来的框是啥样的
    # for box in bboxes2:
    #     small_region = thresh_img[box[1]:box[3], box[0]:box[2]]
    #     black_pixel = len(small_region[small_region == 0])
    #     blank_pixel = len(small_region[small_region == 255])
    #     std_row = np.std(small_region)  # 计算标准差
    #     cv2.rectangle(image_raw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    #     cv2.putText(image_raw, str([black_pixel, blank_pixel, std_row]), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(1) + filename + '.jpg', image_raw)


    # 把上面拿出来的框对应的图像部分抹成白色像素
    # for bbox1 in bboxes2:
    #     image[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]] = 255
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(1) + filename + '.jpg', image)
    # xywh --> xyxy
    # bboxes1 = []
    # for ele in boxes:
    #     x, y, w, h = ele
    #     bboxes1.append([x, y, x+w, y+h])
    # bboxes1_array = np.array(bboxes1)
    # area = (bboxes1_array[:, 3] - bboxes1_array[:, 1]) * (bboxes1_array[:, 2] - bboxes1_array[:, 0])
    # index_ = np.argsort(area)[::-1]    # 从大到小
    # # boxes_ = [bboxes1[ele] for ele in list(index_[:50])]
    #
    # for box in bboxes1:
    #     # x, y, w, h = box
    #     if 0 < abs((box[2] - box[0]) - (box[3] - box[1])) <= 10:
    #         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    #         cv2.putText(image, str([box[0], box[1], box[2], box[3]]), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(1) + filename + '.jpg', image)

    # TODO opencv轮廓线
    # binary = prepare_gray(image1)
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # 这里只找外接矩形

    gray_ = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_, 230, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt_id, cnt in enumerate(reversed(contours)):
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\aa.jpg', image)

    # 框出面积最大的几个就好
    bboxes_array = np.array(bboxes)
    bboxes_area = (bboxes_array[:, 2] - bboxes_array[:, 0]) * (bboxes_array[:, 3] - bboxes_array[:, 1])
    max_index = np.argsort(bboxes_area)[::-1][:8]
    bboxes_new = [bboxes_array[ele] for ele in list(max_index)]
    # 如果框交叉取并集,且去掉框很大那种
    bboxes3 = []
    height, width = image1.shape[:2]
    for ele in bboxes_new:
        width1 = ele[2] - ele[0]
        height1 = ele[3] - ele[1]
        if width1 > int(1/2*width) and width > height:
            continue
        elif height > width and abs(width1 - width) < 30:
            continue
        else:
            bboxes3.append(ele)
    # for ele in bboxes3:
    #     cv2.rectangle(image1, (ele[0], ele[1]), (ele[2], ele[3]), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str('opencv') + filename + '.jpg', image1)

    # 初步拿到一些框,下面判断框相交取最大的框,不能处理的不管,只处理能处理的部分
    # bboxes3 = sorted(bboxes3, key=lambda k: k[0])
    for ele in bboxes3:
        image1[ele[1]:ele[3], ele[0]:ele[2]] = 0

    gray_2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_2, 0, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes4 = []
    for cnt_id, cnt in enumerate(reversed(contours)):
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes4.append((x, y, x + w, y + h))
    for ele in bboxes4:
        cv2.rectangle(image_raw, (ele[0], ele[1]), (ele[2], ele[3]), (0, 0, 255), 1)
    cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str('opencv') + filename + '.jpg', image_raw)
    return image


def image_with_character(image, filename):
    image_raw = image.copy()
    image1 = image.copy()

    gray_ = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_, 230, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt_id, cnt in enumerate(reversed(contours)):
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\aa.jpg', image)

    # 框出面积最大的几个就好
    bboxes_array = np.array(bboxes)
    bboxes_area = (bboxes_array[:, 2] - bboxes_array[:, 0]) * (bboxes_array[:, 3] - bboxes_array[:, 1])
    max_index = np.argsort(bboxes_area)[::-1][:8]
    bboxes_new = [bboxes_array[ele] for ele in list(max_index)]
    # 如果框交叉取并集,且去掉框很大那种
    bboxes3 = []
    height, width = image1.shape[:2]
    for ele in bboxes_new:
        width1 = ele[2] - ele[0]
        height1 = ele[3] - ele[1]
        if width1 > int(1/2*width) and width > height:
            continue
        elif height > width and abs(width1 - width) < 30:
            continue
        else:
            bboxes3.append(ele)
    # for ele in bboxes3:
    #     cv2.rectangle(image1, (ele[0], ele[1]), (ele[2], ele[3]), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str('opencv') + filename + '.jpg', image1)

    # 初步拿到一些框,下面判断框相交取最大的框,不能处理的不管,只处理能处理的部分
    # bboxes3 = sorted(bboxes3, key=lambda k: k[0])
    for ele in bboxes3:
        image1[ele[1]:ele[3], ele[0]:ele[2]] = 0

    gray_2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    ret, binary_ = cv2.threshold(gray_2, 0, 255, cv2.THRESH_BINARY)
    _, contours_, hierarchy = cv2.findContours(binary_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes4 = []
    for cnt_id, cnt in enumerate(reversed(contours_)):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > int(1/2*width) and width > height:
            continue
        elif height > width and abs(w - width) < 30:
            continue
        elif w * h < 20:
            continue
        else:
            bboxes3.append([x, y, x + w, y + h])
        bboxes4.append((x, y, x + w, y + h))
    # for ele in bboxes4:
    #     cv2.rectangle(image_raw, (ele[0], ele[1]), (ele[2], ele[3]), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str('opencv') + filename + '.jpg', image_raw)
    return bboxes4


def erase_character_region(image, filename):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_standard = image / 255
    # grad_X = cv2.Sobel(img_standard, -1, 1, 0)
    # grad_Y = cv2.Sobel(img_standard, -1, 0, 1)
    # grad = cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)

    # grad_X = cv2.Sobel(gray_img, -1, 1, 0)
    # grad_Y = cv2.Sobel(gray_img, -1, 0, 1)
    # sobel = cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)

    sobel = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, ksize=3)
    # canny = cv2.Canny(gray_img, 50, 100)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(00000) + '_canny' + filename, image)
    # _ret, binary = cv2.threshold(sobel, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _ret, binary = cv2.threshold(sobel, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # gray_img = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    # sobel = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, ksize=3)
    # binary = cv2.adaptiveThreshold(sobel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # 设置膨胀和腐蚀操作的核函数
    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    # element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)
    # aim = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,element1, 1 )   #此函数可实现闭运算和开运算
    # 以上膨胀+腐蚀称为闭运算，具有填充白色区域细小黑色空洞、连接近邻物体的作用
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    region = []
    img2, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 利用以上函数可以得到多个轮廓区域，存在一个列表中。
    #  筛选那些面积小的
    for i in range(len(contours)):
        # 遍历所有轮廓
        # cnt是一个点集
        cnt = contours[i]

        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉、这个1000可以按照效果自行设置
        # if (area < 100):
        #     continue

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # 打印出各个矩形四个点的位置
        # print("rect is: ")
        # print(rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if(height > width * 1.3):
            continue

        region.append(box)
    for ele0 in region:
        ele = cv2.boundingRect(ele0)
        xmin1 = ele[0]
        ymin1 = ele[1]
        xmax1 = ele[0] + ele[2]
        ymax1 = ele[1] + ele[3]
        image[ymin1: ymax1, xmin1: xmax1] = 255
        cv2.rectangle(image, (ele[0], ele[1]), (ele[0] + ele[2], ele[1] + ele[3]), (0, 0, 255), 1)
    cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(0) + '_' + filename, image)
    return image


def merge_lines(img_lines, threshold,
                min_line_length=30, max_line_gap=10):
    """
    Merge lines by ends clustering
    """
    raw_lines = cv2.HoughLinesP(img_lines, 1, np.pi / 180, threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
    lines = [sort([(line[0][0], line[0][1]), (line[0][2], line[0][3])]) for line in raw_lines]
    ends = set(flatten(lines))
    ends_map = group_reverse_map(clustering_points(ends, 5))
    merged_set = set([tuple(sort([ends_map[line[0]], ends_map[line[1]]])) for line in lines])
    return [(line[0], line[1]) for line in merged_set]


def cross_point_within_bbox(cross_point, bboxes, image, filename):
    # 找到离box框近的点,去掉无效点
    point_list = []
    # TODO 考虑x,y两个轴,但是轴上的点去不掉
    thresh1 = 10
    # for box in bboxes:
    #     for point in cross_point:
    #         if box[0] + thresh1 <= point[0] <= box[2] - thresh1 and box[1] + thresh1 <= point[1] <= box[3] - thresh1:
    #             point_list.append(point)
    #     else:
    #         continue

    for box in bboxes:
        for point in cross_point:
            if box[0] <= point[0] <= box[0] + thresh1 or box[2] - thresh1 <= point[0] <= box[2]:
                continue
            else:
                point_list.append(point)

    # for ele in point_list:
    #     cv2.circle(image, (ele[0], ele[1]), 30, (255, 0, 255))
    #     cv2.putText(image, str([ele[0], ele[1]]), (int(ele[0]), int(1/2*ele[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str('final') + filename, image)
    # for ele in bboxes:
    #     cv2.rectangle(image, (ele[0], ele[1]), (ele[2], ele[3]), (0, 0, 255), 1)
    #     cv2.putText(image, str([ele[0], ele[1], ele[2], ele[3]]), (int(ele[0]), int(ele[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str('final') + filename, image)
    # point_list_all = [ele for ele in cross_point if ele not in point_list]
    return point_list


def line_detect(image, subject, save_path, filename, file_path):
    s = time.time()
    # 先过ocr将文字部分涂黑   我用的百度ocr,容易发生文字跨栏,势必导致一整片都被抹成白色像素
    # words_result = get_ocr_text_and_coordinate11(image)
    # pattern3 = re.compile('[\u4e00-\u9fa5]|\d+|[A|B|C|D]|[(（]|[)）]|①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩')  # chinese char
    # combine_bbox_list = []
    #
    # for words in words_result:
    #     bbox_s = []
    #     for chars in words['chars']:
    #         char = chars['char']
    #         # pattern_a = re.findall(pattern1, char)
    #         pattern_b = re.findall(pattern3, char)
    #         if len(pattern_b) != 0:
    #             words_xmin = chars['location']['left']
    #             words_ymin = chars['location']['top']
    #             words_xmax = chars['location']['left'] + chars['location']['width']
    #             words_ymax = chars['location']['top'] + chars['location']['height']
    #             bbox_s.append([words_xmin, words_ymin, words_xmax, words_ymax])
    #     if len(bbox_s) != 0:
    #         bbox_s = sorted(bbox_s, key=lambda k: k[0])
    #         bbox_s_array = np.array(bbox_s)
    #         char_w = 20
    #         char_h = 20
    #         bbox_s_xmin = min(bbox_s_array[:, 0]) + int(1/2*char_w)
    #         bbox_s_ymin = min(bbox_s_array[:, 1]) + int(1/2*char_h)
    #         bbox_s_xmax = max(bbox_s_array[:, 2]) - int(1/2*char_w)
    #         bbox_s_ymax = max(bbox_s_array[:, 3]) - int(1/2*char_h)
    #         combine_bbox_list.append([bbox_s_xmin, bbox_s_ymin, bbox_s_xmax, bbox_s_ymax])
    #
    # for bbox1 in combine_bbox_list:
    #     image[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]] = 255
    #     # cv2.rectangle(image, (ele[0], ele[1]), (ele[2], ele[3]), (0, 0, 255), 1)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(0) + '_' + filename, image)

    height, width = image.shape[:2]
    binary = prepare_gray(image)
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt_id, cnt in enumerate(reversed(contours)):
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))

    # 框出面积最大的几个就好
    bboxes_array = np.array(bboxes)
    bboxes_area = (bboxes_array[:, 2] - bboxes_array[:, 0]) * (bboxes_array[:, 3] - bboxes_array[:, 1])
    max_index = np.argsort(bboxes_area)[::-1][:20]
    bboxes_new = [bboxes_array[ele] for ele in list(max_index)]

    image_tmp = image_with_black_pixel(image, bboxes_new)   # 能符合要求的图像很少,很苛刻
    # bboxes_temp = image_with_character(image_tmp, filename)
    # image_1 = erase_character_region(image_new, filename)
    # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(0) + '_' + filename, image_1)

    img_gray = prepare_gray(image_tmp)
    frame_h, frame_v = outline_frame(img_gray, subject)
    point_list1 = cross_points(frame_h, frame_v)

    # 这里还要删掉一部分点,就是字和横线之间形成的点
    # TODO 5-6这部分还是有点问题,需要优化
    # point_list1 = cross_point_within_bbox(point_list1, bboxes_temp, image_tmp, filename)
    # print('cross_point_new:', point_list1)

    # TODO 巨慢,慢到蜗牛爬
    # img_frames = cv2.bitwise_or(frame_h, frame_v)
    # frame_lines = merge_lines(img_frames, 100)
    # cross_point_lines = dict([(point, point_nearby_lines(point, frame_lines, 8)) for point in cross_point])
    # regions = split_region(cross_point, cross_point_lines)
    # img_region = image.copy()
    # for region in regions:
    #     print(region)

    # 删掉一部分点,图像边缘的那些点
    x1 = int(width * 0.01)
    y1 = int(height * 0.01)
    x2 = int(width * 0.99)
    y2 = int(height * 0.99)
    print('cross_point:', point_list1)
    print('image_size', [x1, y1, x2, y2])
    cross_point_list = []
    for index, ele in enumerate(point_list1):
        if x1 <= ele[0] <= x2 and y1 <= ele[1] <= y2:
            cross_point_list.append(ele)
    print('cross_point 个数:', len(cross_point_list))

    # TODO 拿交叉点的方向
    point_list_ = judge_cross_point_and_direction(cross_point_list, image, filename)

    points_list_all = [ele['point'] for ele in point_list_]
    # points_list_cluster = clustering_points(points_list_all, 6)   # 耗时

    points_list_cluster = clustering_points1(points_list_all, 6)   # 耗时

    print('聚类中心的个数:',  len(points_list_cluster.keys()))
    # final_point_list_tmp = align_points(list(points_list_cluster.keys()))
    # 聚类出来的点离原始的点同时平移多少个像素就保留原始点,这个步骤合并在else函数里
    final_point_list = []
    for raw_index, ele in enumerate(points_list_cluster.keys()):
        index_p = 0
        if ele in points_list_all:
            index_p = points_list_all.index(ele)     # 原始点
        else:
            point_x_ = [ele[0] + i for i in range(-10, 10)]
            point_y_ = [ele[1] + i for i in range(-10, 10)]
            point_xy = (list(itertools.product(point_x_, point_y_)))   # 聚类出来的点

            for element in point_xy:
                if element in points_list_all:
                    index_p = points_list_all.index(element)
                    break
        # 聚类出来的点离原始的点同时平移多少个像素就保留原始点,判断聚类出来的点
        point_x_tmp = [ele[0] + i for i in range(-10, 10)]
        point_y_tmp = [ele[1] + i for i in range(-10, 10)]
        point_xy_tmp = (list(itertools.product(point_x_tmp, point_y_tmp)))  # 聚类出来的点
        # point_xy_tmp_array = np.array(point_xy_tmp)

        # raw_point_list = [(ele[0] + i, ele[1] + i) for ele in cross_point_list for i in range(-6, 6)]
        # raw_point_array = np.array(raw_point_list)
        # max_dim = abs(point_xy_tmp_array.shape[0] - raw_point_array.shape[0])
        # if point_xy_tmp_array.shape[0] > raw_point_array.shape[0]:
        #     zero_mat = np.zeros((max_dim, 2), dtype=int)
        #     raw_point_array = np.vstack([raw_point_array, zero_mat])
        # else:
        #     zero_mat = np.zeros((max_dim, 2), dtype=int)
        #     point_xy_tmp_array = np.vstack([point_xy_tmp_array, zero_mat])
        # new_index1 = point_xy_tmp_array[np.logical_and.reduce(point_xy_tmp_array == raw_point_array, axis=1), :]

        # new_index1 = np.arange(0, raw_point_array.shape[0])
        # new_index11 = new_index1[raw_point_array == point_xy_tmp_array]

        # new_index1 = np.array(np.all((point_xy_tmp_array[:, None] == raw_point_array[None, :]), axis=-1).nonzero()).T.tolist()
        # point11 = [raw_point_array[ele[1]] for ele in list(new_index1)]
        for point_t in point_xy_tmp:
            if point_t in cross_point_list:
                if abs(point_t[0] - ele[0]) >= 4 and abs(point_t[1] - ele[1]) >= 4:
                    new_point = tuple(point_t)  # 原始点
                    break
                else:
                    new_point = ele
                    break
        else:
            new_point = ele
        # direction --> four
        direction_temp = point_list_[index_p]['direction']
        DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = False, False, False, False
        if direction_temp[0] == 1:
            DIR_UP = True
        if direction_temp[1] == 1:
            DIR_DOWN = True
        if direction_temp[2] == 1:
            DIR_LEFT = True
        if direction_temp[3] == 1:
            DIR_RIGHT = True
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
        index_true = np.where(np.array(rule_list_tmp) == True)[0]
        if len(index_true) > 2:
            max_index = max(index_true)
            direction = direction_dict[max_index]
        elif len(index_true) == 0:
            continue
        elif len(index_true) == 1:
            direction = direction_dict[index_true[0]]
        else:
            continue
        final_point_dict = {}
        final_point_dict['class_name'] = 'cross_points'
        final_point_dict['bounding_box'] = {'xmin': new_point[0], 'ymin': new_point[1], 'xmax': new_point[0]+1, 'ymax': new_point[1]+1}
        final_point_dict['direction'] = direction   # 自测
        # final_point_dict['direction'] = direction_temp  # 客户端
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

    save_path1 = os.path.join(save_path, subject)
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    # cv2.imwrite(os.path.join(save_path1, file), image)
    write_single_img(image, os.path.join(save_path1, file))
    e = time.time()
    cost_time = e - s
    print('cost_time:', cost_time)


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


def rgb2binary(im):
    gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # _ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    return thresh_img


def get_hist_(image, bboxes_new):
    h_ = []
    for index, box in enumerate(bboxes_new):
        small_image = crop_region_direct(image, box)
        height, width = small_image.shape[:2]
        cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(index) + '_small_image.jpg', small_image)
        h = np.zeros((width, height, 3))
        bins = np.arange(256).reshape(256, 1)  # 直方图中各bin的顶点位置
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR三种颜色
        for ch, col in enumerate(color):
            originHist = cv2.calcHist([small_image], [ch], None, [256], [0, 256])
            cv2.normalize(originHist, originHist, 0, 255 * 0.9, cv2.NORM_MINMAX)
            hist = np.int32(np.around(originHist))
            pts = np.column_stack((bins, hist))
            cv2.polylines(h, [pts], False, col)

        h = np.flipud(h)
        h_.append(h)
    print('h:', h_)
    return h_


def image_with_black_pixel(image, bboxes_new):
    new_box_list = []
    for index, box in enumerate(bboxes_new):
        small_image = crop_region_direct(image, box)
        img_mtx = np.asarray(rgb2binary(small_image))
        y_sum_array = img_mtx.sum(axis=0).sum()
        height, width = small_image.shape[:2]
        # hist = np.histogram(small_image, bins=2)
        # 这个地方想把图像中很小一块地方但是黑色像素很多的地方全部涂黑,
        # 避免后续检测花很久的时间,能涂黑的区域条件还是很苛刻的,一般图像都不会经过这个处理
        if height < 550 and width < 700 and y_sum_array > 12000000:
            # cv2.imwrite(r'E:\111_4_26_test_img\save\\' + str(y_sum_array) + '_small_image.jpg', small_image)
            new_box_list.append(box)
    if len(new_box_list) == 1:
        bbox_ = new_box_list[0]
        image[bbox_[1]: bbox_[3], bbox_[0]: bbox_[2]] = 255
        # cv2.imwrite(r'E:\111_4_26_test_img\save\new_img.jpg', image)
        return image
    else:
        return image


if __name__ == '__main__':
    # img_path = r'E:\111_4_26_test_img\aa\a\1.jpg'
    # image = cv2.imread(img_path)
    # image1 = crop_region_direct(image, [2317, 1, 4430, 2936])
    # line_detect(image)

    img_path = r'E:\111_4_26_test_img\images'
    save_path = r'E:\111_4_26_test_img\save2'
    for root, dirs, files in os.walk(img_path):
        for file in files:
            file_path = os.path.join(root, file)
            subject = file_path.split('\\')[-2]
            # image = read_single_img(file_path)
            img = Image.open(file_path)
            print(file_path)
            img_pil, image = convert_pil_to_jpeg(img)
            img_nostamp = remove_stamp(image)
            # s = time.time()
            line_detect(image, subject, save_path, file, file_path)
            # e = time.time()
            # cost_time = e - s
            # print('cost_time:', cost_time)