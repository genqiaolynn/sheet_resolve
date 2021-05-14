# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/5/12 18:12'

import numpy as np
import cv2


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


def merge_lines(img_lines, threshold,
                min_line_length=30, max_line_gap=10):
    """
    Merge lines by ends clustering
    """
    # raw_lines = cv2.HoughLinesP(img_lines, 1, np.pi / 180, threshold,
    #                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    raw_lines = cv2.HoughLinesP(img_lines, 1, np.pi / 180, 160, minLineLength=500, maxLineGap=65)

    lines = [sort([(line[0][0], line[0][1]), (line[0][2], line[0][3])]) for line in raw_lines]
    ends = set(flatten(lines))
    ends_map = group_reverse_map(clustering_points(ends, 5))
    merged_set = set([tuple(sort([ends_map[line[0]], ends_map[line[1]]])) for line in lines])
    return [(line[0], line[1]) for line in merged_set]


def merge_lines1(lines):
    lines = [sort([(line[0], line[1]), (line[2], line[3])]) for line in lines]
    ends = set(flatten(lines))
    ends_map = group_reverse_map(clustering_points(ends, 3))
    merged_set = set([tuple(sort([ends_map[line[0]], ends_map[line[1]]])) for line in lines])
    return [(line[0], line[1]) for line in merged_set]


def detect_lines(image):
    img_gray = prepare_gray(image)
    frame_h, frame_v = outline_frame(img_gray, 6)
    img_frames = cv2.bitwise_or(frame_h, frame_v)
    # frame_lines = merge_lines(img_frames, 20)

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fld = cv2.ximgproc.createFastLineDetector()
    dlines = fld.detect(binary)
    lines = [line[0] for line in dlines.tolist()]


    lines_arr = np.array(lines)
    width_ = lines_arr[:, 2] - lines_arr[:, 0]
    height_ = lines_arr[:, 3] - lines_arr[:, 1]
    lines_index1 = np.where(width_ >= 20)[0]
    lines_index2 = np.where(height_ >= 20)[0]
    lines_index = np.hstack([lines_index1, lines_index2])
    new_lines = [lines[ele] for ele in lines_index]


    frame_lines = merge_lines1(new_lines)
    # img_lines = np.zeros(image.shape)
    # for line in frame_lines:
    #     cv2.line(img_lines, tuple(map(int, line[0])), tuple(map(int, line[1])), (255, 255, 255), thickness=1)
    # cv2.imwrite(r'E:\111_4_26_test_img\\' + str(0) + '_' + '2.jpg', img_lines)


if __name__ == '__main__':
    img_path = r'E:\111_4_26_test_img\1.jpg'
    image = cv2.imread(img_path)
    detect_lines(image)