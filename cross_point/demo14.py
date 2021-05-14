# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/5/6 11:12'


import cv2, os, shutil
import numpy as np
import xml.etree.ElementTree as ET
from utils import *
from PIL import Image


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


def group_reverse_map(group_res, value=lambda v: v, key=lambda k: k):
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


# 自适应阈值出来了很多噪点,感觉更适合自然场景多些
def prepare_gray_(img_color):
    img_gray = cv2.bitwise_not(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY))
    return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)


def prepare_gray(img_color):
    gray_img = cv2.bitwise_not(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY))
    # return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    _ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_img


# def outline_frame(img_gray, border_thickness, horizontal_scale=20.0, vertical_scale=20.0):
def outline_frame(img_gray, border_thickness, subject, horizontal_scale=30.0, vertical_scale=30.0):
    # 语文这个阈值调小点25,因为作文好多线   其他学科暂时觉得越大越好,能找到足够多的点再去删
    if subject == '语文':
        horizontal_scale = 20
        vertical_scale = 20
    else:
        horizontal_scale = 45
        vertical_scale = 45
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
    return align_points(list(clustering_points(zip(cross_xs, cross_ys), 5).keys()))


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


def judge_cross_point_and_direction(points_list, binary, name, thresh=9, pixel_thresh=0):
    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite(r'E:\111_4_26_test_img\aa\222\thresh_img.jpg', thresh_img)

    cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + 'ada_box.jpg', binary)
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
        cv2.imwrite(r'E:\111_4_26_test_img\save\\' + name + '_box.jpg', binary)
    return points_list_tmp


def line_detect(image, subject, save_path, filename, file_path):
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

    image_new = get_hist(image, bboxes_new)
    img_gray = prepare_gray(image_new)
    frame_h, frame_v = outline_frame(img_gray, 4, subject)
    cross_point = cross_points(frame_h, frame_v)

    # 删掉一部分点,图像边缘的那些点
    x1 = int(width * 0.05)
    y1 = int(height * 0.05)
    x2 = int(width * 0.95)
    y2 = int(height * 0.95)
    print('cross_point:', cross_point)
    print('image_size', [x1, y1, x2, y2])
    cross_point_list = []
    for index, ele in enumerate(cross_point):
        if x1 <= ele[0] <= x2 and y1 <= ele[1] <= y2:
            cross_point_list.append(ele)
    print('cross_point 个数:', len(cross_point_list))

    # TODO 拿交叉点的方向
    # point_list_ = judge_cross_point_and_direction(cross_point_list, binary, filename, thresh=9)
    # print(point_list_)

    # TODO 加方向画点
    # template = r'./exam_info/000000-template.xml'
    # tree = ET.parse(template)
    # for points in point_list_:
    #     points2 = points['point']
    #     create_xml(str(points2) + '_' + str(points['std']) + '_' + points['direction'], tree, int(points2[0]), int(points2[1]), int(points2[0] + 1), int(points2[1] + 1))
    # save_path1 = os.path.join(save_path, subject)
    # if not os.path.exists(save_path1):
    #     os.makedirs(save_path1)
    # tree.write(os.path.join(save_path1, file.replace('.jpg', '.xml')))
    # shutil.copyfile(file_path, os.path.join(save_path1, file))

    # cv2.imwrite(os.path.join(save_path1, file), image)
    # write_single_img(image, save_path1)


    # TODO 不加方向画点
    template = r'./exam_info/000000-template.xml'
    tree = ET.parse(template)
    for points2 in cross_point_list:
        create_xml(str(points2) + '_', tree, int(points2[0]), int(points2[1]), int(points2[0] + 1), int(points2[1] + 1))
    save_path1 = os.path.join(save_path, subject)
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    tree.write(os.path.join(save_path1, file.replace('.jpg', '.xml')))
    shutil.copyfile(file_path, os.path.join(save_path1, file))


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


def get_hist(image, bboxes_new):
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


def get_chart_pic_from_image_(image):
    height, width = image.shape[:2]
    ratio = 1
    resize_image = cv2.resize(image, (int(width/ratio), int(height/ratio)), cv2.INTER_NEAREST)
    img_gray = prepare_gray(resize_image)
    _, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    bboxes = []
    for cnt_id, cnt in enumerate(reversed(contours)):
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))

    # 框出面积最大的几个就好
    bboxes_array = np.array(bboxes)
    bboxes_area = (bboxes_array[:, 2] - bboxes_array[:, 0]) * (bboxes_array[:, 3] - bboxes_array[:, 1])
    max_index = np.argsort(bboxes_area)[::-1][:20]
    bboxes_new = [bboxes_array[ele] for ele in list(max_index)]
    for pixel in bboxes_new:
        x_min, y_min, x_max, y_max = pixel
        cv2.rectangle(image, (int(x_min * ratio), int(y_min * ratio)), (int(x_max * ratio), int(y_max * ratio)), (255, 0, 255), 1)
        cv2.imwrite(r'E:\111_4_26_test_img\save\save.jpg', image)
    h = get_hist(image, bboxes_new)
    print('h:', h)

    # frame_h, frame_v = outline_frame(img_gray, 4)
    #
    # fld = cv2.ximgproc.createFastLineDetector()
    # dlines = fld.detect(img_gray)
    # lines = [line[0] for line in dlines.tolist()]
    #
    # lines_arr = np.array(lines)
    # width_ = lines_arr[:, 2] - lines_arr[:, 0]
    # height_ = lines_arr[:, 3] - lines_arr[:, 1]
    # lines_index1 = np.where(width_ > 50)[0]
    # lines_index2 = np.where(height_ > 50)[0]
    # lines_index = np.hstack([lines_index1, lines_index2])
    # new_lines = [lines[ele] for ele in lines_index]
    # print(new_lines)


def get_chart_pic_from_image(image):
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
    # for pixel in bboxes_new:
    #     x_min, y_min, x_max, y_max = pixel
    #     cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 1)
    #     cv2.imwrite(r'E:\111_4_26_test_img\save\save.jpg', image)
    image_new = get_hist(image, bboxes_new)
    binary_new = prepare_gray(image_new)


if __name__ == '__main__':
    # img_path = r'E:\111_4_26_test_img\aa\a\1.jpg'
    # image = cv2.imread(img_path)
    # image1 = crop_region_direct(image, [2317, 1, 4430, 2936])
    # line_detect(image)

    img_path = r'E:\111_4_26_test_img\images'
    save_path = r'E:\111_4_26_test_img\save'
    for root, dirs, files in os.walk(img_path):
        for file in files:
            file_path = os.path.join(root, file)
            subject = file_path.split('\\')[-2]
            # image = read_single_img(file_path)
            img = Image.open(file_path)
            print(file_path)
            img_pil, image = convert_pil_to_jpeg(img)
            line_detect(image, subject, save_path, file, file_path)
