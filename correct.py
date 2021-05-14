# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/4/26 19:42'


import numpy as np
import cv2
from numpy import asarray
import base64
import scipy.signal

import utils


def hough_rotate_cv(image):
    """ not Long time consuming, not Strong generalization ability, not high accuracy, more super parameters"""
    img_np = utils.resize_by_percent(asarray(image), 1)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(img_np, 0, 255, apertureSize=3)
    # cv2.imshow('canny', canny_image)
    # cv2.waitKey(10)
    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 160, minLineLength=500, maxLineGap=65)
    # lines = cv2.HoughLines(canny_image, 1, np.pi / 180, 160, max_theta=30, min_theta=0)

    # 寻找长度最长的线
    distance = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dis = np.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
        distance.append(dis)
    max_dis_index = distance.index(max(distance))
    max_line = lines[max_dis_index]
    x1, y1, x2, y2 = max_line[0]

    # 获取旋转角度
    angle = cv2.fastAtan2((y2 - y1), (x2 - x1))
    print(angle)

    if 0.5 <= angle <= 7:  # 因为识别误差问题，根据实际情况设置旋转阈值
        centerpoint = (image.shape[1] / 2, image.shape[0] / 2)
        rotate_mat = cv2.getRotationMatrix2D(centerpoint, angle, 1.0)  # 获取旋转矩阵
        correct_image = cv2.warpAffine(image, rotate_mat, (image.shape[1], image.shape[0]),
                                       borderValue=(255, 255, 255))

        # cv2.imshow('test', resize_by_percent(correct_image, 0.1))
        # cv2.waitKey(10)
        return correct_image
    else:
        return image


def array_latter_subtracts_precious(nparray):
    array1 = nparray[:-1]
    array2 = nparray[1:]
    return array2 - array1


def split_by_index(im_raw, index):
    y_raw, x_raw, _ = im_raw.shape
    img_left = im_raw[1:y_raw, 1:index]
    img_right = im_raw[1:y_raw, index + 1:x_raw]
    return img_left, img_right


def split_img_at_middle_by_y_axis(img_path, radio=0.10, thresh_std=5000):
    im_raw = utils.read_img(img_path)
    im_resize = utils.resize_by_percent(im_raw, radio)
    ry, rx, _ = im_resize.shape
    img_mtx0 = np.asarray(utils.rgb2binary(im_resize))
    y_sum_array0 = img_mtx0.sum(axis=0)
    tmp = array_latter_subtracts_precious(y_sum_array0 / ry)
    std0 = np.std(tmp)  # 计算标准差

    # # plt.bar(range(len(y_sum_array0)), y_sum_array0)
    # # plt.show()
    # plt.plot(range(len(y_sum_array0)-1), tmp)
    # plt.show()

    y, x, _z = im_resize.shape
    x_bias = int(x * 0.15)
    y_bias = int(y * 0.30)
    middle_x = int(x / 2)
    middle_area_img = im_resize[y_bias:y, middle_x - x_bias:middle_x + x_bias]
    img_mtx = np.asarray(utils.rgb2binary(middle_area_img))
    y_sum_array = img_mtx.sum(axis=0)
    std = np.std(y_sum_array)  # 计算标准差
    y_sum_list = list(y_sum_array)

    if std <= thresh_std:
        index = y_sum_list.index(max(y_sum_list))
    else:
        index = y_sum_list.index(min(y_sum_list))
    split_index = middle_x + index - int(len(y_sum_list) / 2)
    split_index = int(split_index / radio)

    y_raw, x_raw, _ = im_raw.shape
    img_left = im_raw[1:y_raw, 1:split_index]
    img_right = im_raw[1:y_raw, split_index + 1:x_raw]
    left_path = img_path.replace('.jpg', '_left.jpg')
    right_path = img_path.replace('.jpg', '_right.jpg')
    cv2.imencode('.jpg', img_left)[1].tofile(left_path)
    cv2.imencode('.jpg', img_right)[1].tofile(right_path)
    print(left_path)
    print(right_path)


def smart_split_img_at_middle_by_x_axis(img_path, resize_radio=0.1):
    im_raw = utils.read_img(img_path)
    im_resize = utils.resize_by_percent(im_raw, resize_radio)

    bin_img = utils.rgb2binary(im_resize)
    ry, rx = bin_img.shape
    img_mtx0 = np.asarray(bin_img)
    y_sum_array0 = img_mtx0.sum(axis=0)  # y轴求和
    subtracts_arr = np.abs(array_latter_subtracts_precious(y_sum_array0 / ry))  # 长度减1
    subtracts_arr_index = np.argsort(subtracts_arr, kind='quicksort', order=None)
    subtracts_arr_index = subtracts_arr_index[-10:]

    index_middle_distance_list = list(np.abs(subtracts_arr_index - int(rx / 2)))
    split_index = subtracts_arr_index[index_middle_distance_list.index(min(index_middle_distance_list))] + 1
    split_index = int(split_index / resize_radio)
    img_left, img_right = split_by_index(im_raw, split_index)
    left_path = img_path.replace('.jpg', '_left.jpg')
    right_path = img_path.replace('.jpg', '_right.jpg')
    cv2.imencode('.jpg', img_left)[1].tofile(left_path)
    cv2.imencode('.jpg', img_right)[1].tofile(right_path)
    print(left_path)
    print(right_path)


def segment2parts_by_pix(crop_img):

    p_image = utils.preprocess(crop_img)
    height, width = p_image.shape
    sum_x_axis = p_image.sum(axis=0) / (height*255)

    # sum_x_axis = (sum_x_axis / (255*height)).astype(float)
    kernel = np.array([-2, 0, 2])
    sobel_filter = scipy.signal.convolve(sum_x_axis, kernel)  # 一维卷积运算

    temp = np.abs(sobel_filter[1:-1])/np.max(np.abs(sobel_filter[1:-1]))
    temp[temp < 0.6] = 0
    temp[temp != 0] = 1
    index = np.where(temp == 1)[0]

    width1 = width // 9

    intervals = [(0, width1), (4 * width1, 5 * width1), (8 * width1, width)]  # 左开右闭

    index_list = []
    for i, interval in enumerate(intervals):
        index_sec_list = []
        for ele in index:
            if interval[0] < ele <= interval[1]:
                index_sec_list.append(ele)

        index_list.append(index_sec_list)

    left_x_point, middle_x_point, right_x_point = 9999, 9999, 9999
    left_del_part = (0, left_x_point)
    middle_part = (left_x_point, middle_x_point)
    right_part = (middle_x_point, right_x_point)
    right_del_part = (right_x_point, width)

    # left
    if index_list[0]:
        left_x_point = index_list[0][-1]
        left_del_part = (0, left_x_point)
    # middle
    if index_list[1]:
        value_list = [abs(sobel_filter[index]) for index in index_list[1]]
        middle_x_point = index_list[1][value_list.index(max(value_list))]
        middle_part = (left_x_point, middle_x_point)
    # right
    if index_list[2]:
        right_x_point = index_list[2][0]
        right_part = (middle_x_point, right_x_point)
        right_del_part = (right_x_point, width)

    split_point = sorted(list(set(sorted(list(left_del_part + middle_part + right_part + right_del_part))) - {9999}))

    split_pairs = []
    if len(split_point) > 2:
        a = split_point[:-1]
        b = split_point[1:]
        for i, ele in enumerate(a):
            if b[i] - ele > width1:
                split_pairs.append((ele, b[i]))

    return split_pairs


def segment2parts(im_raw, save_path):
    img_parts_dict_list = []

    # randon_img = radon_rotate_ski(im_raw)
    # 试卷顶部可能有黑边，切去3%
    yy, xx = im_raw.shape[0], im_raw.shape[1]
    y_crop_pix = int(yy*0.03)
    # x_crop_pix = int(xx*0.03)
    x_crop_pix = 0
    im_crop = im_raw[y_crop_pix:yy-y_crop_pix, x_crop_pix:xx-x_crop_pix]

    split_pairs = segment2parts_by_pix(im_crop)
    if len(split_pairs) >= 2:
        for index, ele in enumerate(split_pairs):
            dst = im_raw[:, ele[0]:ele[1]]
            save_path_final = save_path.replace('.jpg', '') + '_{}_{}_{}.jpg'.format(ele[0], 0, index)
            cv2.imencode('.jpg', dst)[1].tofile(save_path_final)
            image = cv2.imencode('.jpg', dst)[1]
            base64_data = str(base64.b64encode(image))[2:-1]
            part_dict = {'img_part': base64_data,
                         'x_bias': ele[0] + x_crop_pix,
                         'y_bias': 0}

            img_parts_dict_list.append(part_dict)

    else:
        img = im_crop[:, split_pairs[0][0]:split_pairs[0][1]]
        resize_ratio = 0.3
        im_resize = utils.resize_by_percent(img, resize_ratio)

        # gray
        if len(im_resize.shape) >= 3:
            gray_img = cv2.cvtColor(im_resize, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = im_resize
        ry, rx = gray_img.shape
        # 高斯
        glur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        # otsu
        _ret, threshed_img = cv2.threshold(glur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if ry < rx:
            x_kernel = int(10*resize_ratio)
        else:
            x_kernel = int(10 * resize_ratio)
        kernel = np.ones((glur_img.shape[0], x_kernel), np.uint8)  # height, width
        dilation = cv2.dilate(threshed_img, kernel, iterations=1)
        # cv2.imshow(' ', dilation)
        # if cv2.waitKey(0) == 27:
        #     cv2.destroyAllWindows()

        # _, cnts, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (major, minor, _) = cv2.__version__.split(".")
        contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours[0] if int(major) > 3 else contours[1]

        box_list = [cv2.boundingRect(cnt) for cnt in cnts]
        box_array = np.asarray(box_list)
        box_array[:, 2] = box_array[:, 0] + box_array[:, 2]
        box_array[:, 3] = box_array[:, 1] + box_array[:, 3]

        middle_x = rx // 2
        left_box = np.asarray([0, 0, 0, 0])
        right_box = np.asarray([0, 0, 0, 0])
        for box in box_array:
            x, y, xmax, ymax = box
            if x + (xmax-x)//2 <= middle_x:
                left_box = np.vstack([left_box, box])
            else:
                right_box = np.vstack([right_box, box])

        left_box_list = []
        right_box_list = []
        try:
            left_box_list = left_box[1:, :][:, :2].min(axis=0).tolist() + left_box[1:, :][:, 2:].max(axis=0).tolist()
        except Exception:
            pass  # 单面的情况
        try:
            right_box_list = right_box[1:, :][:, :2].min(axis=0).tolist() + right_box[1:, :][:, 2:].max(axis=0).tolist()
        except Exception:
            pass

        box_list = [left_box_list, right_box_list]

        bias = int(70 * resize_ratio)
        for index, box in enumerate(box_list):
            if len(box) > 0:
                xmin, ymin, xmax, ymax = box
                if xmin - bias > 0:
                    xmin = xmin - bias
                else:
                    xmin = 0

                dst = im_crop[int(ymin / resize_ratio):int(ymax / resize_ratio),
                      int(xmin / resize_ratio):int(xmax / resize_ratio)]
                save_path_final = save_path.replace('.jpg', '') + '_{}_{}_{}.jpg'.format(xmin, ymin, index)
                cv2.imencode('.jpg', dst)[1].tofile(save_path_final)
                image = cv2.imencode('.jpg', dst)[1]
                base64_data = str(base64.b64encode(image))[2:-1]
                part_dict = {'img_part': base64_data,
                             'x_bias': int(xmin/resize_ratio) + x_crop_pix + split_pairs[0][0],
                             'y_bias': int(ymin/resize_ratio) + y_crop_pix + 0}
                if (xmax - xmin)/resize_ratio > 100:  # 去掉竖长条
                    img_parts_dict_list.append(part_dict)

    return img_parts_dict_list

