# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/22 10:29'


import traceback
import brain_api_charge as brain_api
from itertools import chain
import re
import numpy as np
import cv2
import utils
import xml.etree.cElementTree as ET
from utils import crop_region_direct, create_xml, infer_number, combine_char_in_raw_format
from sklearn.cluster import DBSCAN
from ocr2sheet_demo import ocr2sheet


def get_split_index(array, dif=0):
    array = np.array(array)
    interval_list = np.abs(array[1:] - array[:-1])
    split_index = [0]
    for i, interval in enumerate(interval_list):
        if dif:
            split_dif = dif
        else:
            split_dif = np.mean(interval_list)
        if interval > split_dif:
            split_index.append(i + 1)

    split_index.append(len(array))
    split_index = sorted(list(set(split_index)))
    return split_index


def adjust_choice_m(image, xe, ye):
    dilate = 1
    blur = 5

    # Convert to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur != 0:
        image = cv2.GaussianBlur(image, (blur, blur), 0)

    # Apply threshold to get image with only b&w (binarization)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = np.ones((ye, xe), np.uint8)  # y轴膨胀, x轴膨胀

    dst = cv2.dilate(image, kernel, iterations=1)

    (major, minor, _) = cv2.__version__.split(".")
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours[0] if int(major) > 3 else contours[1]

    # _, cnts, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    right_limit = 0
    bottom_limit = 0
    for cnt_id, cnt in enumerate(reversed(cnts)):
        x, y, w, h = cv2.boundingRect(cnt)
        if x + w > right_limit:
            right_limit = x + w

        if y + h > bottom_limit:
            bottom_limit = y + h

    return right_limit, bottom_limit


def find_digital(ocr_raw_list, left, top):
    pattern = r'\d+'
    x_list = []
    y_list = []
    digital_list = list()
    chars_list = list()
    height_list, width_list = list(), list()
    ocr_dict_list = combine_char_in_raw_format(ocr_raw_list, left, top)
    for i, ele in enumerate(ocr_dict_list):
        words = ele['words']
        words = words.replace(' ', '').upper()  # 去除空格

        digital_words_m = re.finditer(pattern, words)
        digital_index_list = [(m.group(), m.span()) for m in digital_words_m if m]
        chars_index = [ele for ele in range(0, len(ele['chars']))]
        digital_index_detail_list = []
        for letter_info in digital_index_list:
            number = letter_info[0]
            index_start = letter_info[1][0]
            index_end = letter_info[1][1] - 1
            char_start = ele['chars'][index_start]
            char_end = ele['chars'][index_end]

            if index_start == index_end:
                digital_index_detail_list += [index_start]
            else:
                digital_index_detail_list += chars_index[index_start:index_end + 1]

            letter_loc_xmin = int(char_start['location']['left'])
            letter_loc_ymin = min(int(char_start['location']['top']), int(char_end['location']['top']))
            letter_loc_xmax = int(char_end['location']['left']) + int(char_end['location']['width'])
            letter_loc_ymax = max(int(char_start['location']['top']) + int(char_start['location']['height']),
                                  int(char_end['location']['top']) + int(char_end['location']['height']))

            mid_x = letter_loc_xmin + (letter_loc_xmax - letter_loc_xmin) // 2
            mid_y = letter_loc_ymin + (letter_loc_ymax - letter_loc_ymin) // 2

            # print(number, (mid_x, mid_y))
            x_list.append(mid_x)
            y_list.append(mid_y)

            height_list.append(letter_loc_ymax - letter_loc_ymin)
            width_list.append(letter_loc_xmax - letter_loc_xmin)

            number_loc = (letter_loc_xmin, letter_loc_ymin, letter_loc_xmax, letter_loc_ymax, mid_x, mid_y)
            digital_list.append({"digital": int(number), "loc": number_loc})

        current_chars = [char for index, char in enumerate(ele['chars'])
                         if index not in digital_index_detail_list and char['char'].encode('utf-8').isalpha()]

        chars_list += current_chars

    if not height_list or not width_list:
        d_mean_height, d_mean_width = 0, 0
    else:
        d_mean_height = sum(height_list) // len(height_list)
        d_mean_width = sum(width_list) // len(width_list)

    # mean_height = max(height_list)
    # mean_width = max(width_list)
    # print(x_list)
    # print(y_list)
    return digital_list, chars_list, d_mean_height, d_mean_width


def cluster2choice_m_(cluster_list, m_h, m_w):
    numbers = [ele['digital'] for ele in cluster_list]

    loc_top_interval = (np.array([ele['loc'][3] for ele in cluster_list][1:]) -
                        np.array([ele['loc'][3] for ele in cluster_list][:-1]))

    split_index = [0]
    for i, interval in enumerate(loc_top_interval):
        if interval > m_h * 1.5:
            split_index.append(i + 1)

    split_index.append(len(cluster_list))
    split_index = sorted(list(set(split_index)))
    block_list = []
    for i in range(len(split_index) - 1):
        block = cluster_list[split_index[i]: split_index[i + 1]]

        xmin = min([ele["loc"][0] for ele in block])
        ymin = min([ele["loc"][1] for ele in block])
        xmax = max([ele["loc"][2] for ele in block])
        ymax = max([ele["loc"][3] for ele in block])

        numbers = [ele['digital'] for ele in block]

        choice_m = {"number": numbers, "loc": (xmin, ymin, xmax, ymax)}
        block_list.append(choice_m)

    return block_list


def cluster2choice_m(cluster_list, mean_width, tf_box=False):
    # 比较x坐标，去掉误差值
    numbers_x = [ele['loc'][4] for ele in cluster_list]
    numbers_x_array = np.array(numbers_x)
    numbers_x_interval = np.abs((numbers_x_array[1:] - numbers_x_array[:-1]))
    error_index_superset = np.where(numbers_x_interval >= mean_width)[0]
    error_index_superset_interval = error_index_superset[1:] - error_index_superset[:-1]
    t_index = list(np.where(error_index_superset_interval > 1)[0] + 1)
    t_index.insert(0, 0)
    t_index.append(len(error_index_superset))
    error = []
    for i in range(0, len(t_index) - 1):
        a = t_index[i]
        b = t_index[i + 1]
        block = list(error_index_superset[a: b])
        error += block[1:]

    cluster_list = [ele for i, ele in enumerate(cluster_list) if i not in error]
    numbers = [ele['digital'] for ele in cluster_list]
    numbers_array = np.array(numbers)

    # numbers_y = [ele['loc'][5] for ele in cluster_list]
    # numbers_y_array = np.array(numbers_y)
    # numbers_y_interval = np.abs((numbers_y_array[1:] - numbers_y_array[:-1]))
    # split_index = [0]
    # for i, interval in enumerate(numbers_y_interval):
    #     if interval > np.mean(numbers_y_interval):
    #         split_index.append(i + 1)
    #
    # split_index.append(len(cluster_list))
    # split_index = sorted(list(set(split_index)))
    # for i in range(len(split_index) - 1):
    #     block = cluster_list[split_index[i]: split_index[i + 1]]
    #     block_numbers = numbers_array[split_index[i]: split_index[i + 1]]

    # 确定数字题号的位置，前提：同block题号是某等差数列的子集
    numbers_sum = numbers_array + np.flipud(numbers_array)

    counts = np.bincount(numbers_sum)
    mode_times = np.max(counts)
    mode_value = np.argmax(counts)

    if mode_times != len(numbers) and mode_times >= 2:
        # 启动题号补全

        number_interval_list = abs(numbers_array[1:] - numbers_array[:-1])
        number_interval_counts = np.bincount(number_interval_list)
        # number_interval_mode_times = np.max(number_interval_counts)
        number_interval_mode_value = np.argmax(number_interval_counts)

        suspect_index = np.where(numbers_sum != mode_value)[0]
        numbers_array_len = len(numbers_array)
        for suspect in suspect_index:
            if suspect == 0:
                cond_left = False
                cond_right = numbers_array[suspect + 1] == numbers_array[suspect] + number_interval_mode_value
            elif suspect == numbers_array_len - 1:
                cond_right = False
                cond_left = numbers_array[suspect - 1] == numbers_array[suspect] - number_interval_mode_value
            else:
                cond_left = numbers_array[suspect - 1] == numbers_array[suspect] - number_interval_mode_value
                cond_right = numbers_array[suspect + 1] == numbers_array[suspect] + number_interval_mode_value

            # if cond_right and not cond_left and suspect != numbers_array_len - 1:
            #    numbers_array[suspect] = -1

            # 以下是将错的纠正但是不补数字
            if cond_left or cond_right:
                pass
            else:
                numbers_array[suspect] = -1

        times = 0
        numbers_array = infer_number(numbers_array, times, number_interval_mode_value)  # 推断题号
        numbers_array = np.array(numbers_array)

    numbers_interval = np.abs(numbers_array[1:] - numbers_array[:-1])

    split_index = [0]
    for i, interval in enumerate(numbers_interval):
        if interval > np.mean(numbers_interval):
            split_index.append(i + 1)

    split_index.append(len(cluster_list))
    split_index = sorted(list(set(split_index)))
    block_list = []

    if tf_box:
        split_index = [0, len(cluster_list)]

    for i in range(len(split_index) - 1):
        block = cluster_list[split_index[i]: split_index[i + 1]]
        block_numbers = numbers_array[split_index[i]: split_index[i + 1]]

        xmin = min([ele["loc"][0] for ele in block])
        ymin = min([ele["loc"][1] for ele in block])
        xmax = max([ele["loc"][2] for ele in block])
        ymax = max([ele["loc"][3] for ele in block])
        mid_x = xmin + (xmax - xmin) // 2
        mid_y = ymin + (ymax - ymin) // 2

        choice_m = {"numbers": list(block_numbers), "loc": [xmin, ymin, xmax, ymax, mid_x, mid_y]}
        block_list.append(choice_m)

    return block_list


# polygon 多边形
def point_in_polygon(point, polygon):
    xmin, ymin, xmax, ymax = polygon['xmin'], polygon['ymin'], polygon['xmax'], polygon['ymax']
    if xmin <= point[0] <= xmax and ymin <= point[1] <= ymax:
        return True
    else:
        return False


def cluster_and_anti_abnormal(image, xml_path, choice_n_list, digital_list, chars_list,
                              mean_height, mean_width, choice_s_height, choice_s_width, limit_loc):
    limit_left, limit_top, limit_right, limit_bottom = limit_loc
    limit_width, limit_height = limit_right - limit_left, limit_bottom - limit_top
    digital_loc_arr = []
    digital_list_to_cluster = []
    # 在choice_n 的数字不进行聚类
    for i, ele in enumerate(digital_list):
        point = [ele["loc"][-2], ele["loc"][-1]]
        contain = False
        for choice_n in choice_n_list:
            choice_n_loc = choice_n['bounding_box']

            numbers = choice_n.get('numbers')
            if not numbers:
                choice_n.update({"numbers": []})
            if point_in_polygon(point, choice_n_loc):
                contain = True
                choice_n["numbers"].append(digital_list[i])
                break
        if not contain:
            digital_list_to_cluster.append(digital_list[i])
            digital_loc_arr.append(point)

    # 得到所有题号区域， 作为后续划分choice_m的依据
    choice_m_numbers_list = []
    for ele in choice_n_list:
        loc = ele['bounding_box']
        xmin, ymin, xmax, ymax = loc['xmin'], loc['ymin'], loc['xmax'], loc['ymax']
        mid_x, mid_y = (xmax - xmin) // 2 + xmin, (ymax - ymin) // 2 + ymin

        cluster = ele.get('numbers')
        if not cluster:
            block_list = [{"numbers": [-1], "loc": [xmin, ymin, xmax, ymax, mid_x, mid_y]}]
        else:
            block_list = cluster2choice_m(cluster, mean_width, tf_box=True)
            block_list[0].update({"loc": [xmin, ymin, xmax, ymax, mid_x, mid_y]})

        choice_m_numbers_list += block_list

    if digital_loc_arr:
        digital_loc_arr = np.array(digital_loc_arr)
        if choice_s_height != 0:
            eps = int(choice_s_height * 2.5)
        else:
            eps = int(mean_height * 3)
        print("eps: ", eps)
        db = DBSCAN(eps=eps, min_samples=2, metric='chebyshev').fit(digital_loc_arr)
        # 默认的metric是欧式距离，chebyshev是chebyshev
        # 密度聚类之DBSCAN算法需要两个参数：ε(eps) ε-邻域(这个参数是设定的半径r)和形成高密度区域所需要的最少点数(minPts)
        # https://zhuanlan.zhihu.com/p/54833132         这个链接讲的原理知识
        # https://www.cntofu.com/book/85/ml/cluster/scikit-learn-dbscan.md    官方文档
        labels = db.labels_
        print(labels)

        cluster_label = []
        for ele in labels:
            if ele not in cluster_label and ele != -1:
                cluster_label.append(ele)

        a_e_dict = {k: [] for k in cluster_label}

        for index, ele in enumerate(labels):
            if ele != -1:
                a_e_dict[ele].append(digital_list_to_cluster[index])

        for ele in cluster_label:
            cluster = a_e_dict[ele]
            choice_m_numbers_list += cluster2choice_m(cluster, mean_width)

    all_list_nums = [ele["numbers"] for ele in choice_m_numbers_list]
    all_nums_len = [len(ele) for ele in all_list_nums]
    all_nums = list(chain.from_iterable(all_list_nums))

    # counts = np.bincount(np.array(all_nums_len))
    # if np.max(counts) < 2:
    #     mode_value = max(all_nums_len)
    # else:
    #     mode_value = np.argmax(counts)
    #     mode_value = all_nums_len[np.where(np.array(all_nums_len) == mode_value)[0][-1]]
    #
    # if mode_value > 1:  # 缺失补全
    #     error_index_list = list(np.where(np.array(all_nums_len) != mode_value)[0])
    #
    #     all_height = [ele["loc"][3] - ele["loc"][1] for index, ele
    #                   in enumerate(choice_m_numbers_list) if index not in error_index_list]
    #     choice_m_mean_height = int(sum(all_height) / len(all_height))
    #
    #     for e_index in list(error_index_list):
    #         current_choice_m = choice_m_numbers_list[e_index]
    #         current_numbers_list = list(all_list_nums[e_index])
    #         current_len = all_nums_len[e_index]
    #         dif = mode_value - current_len
    #
    #         if 1 in current_numbers_list:
    #             t2 = current_numbers_list + [-1] * dif
    #             infer_t1_list = infer_number(t2)  # 后补
    #             infer_t2_list = infer_number(t2)  # 后补
    #             cond1 = False
    #             cond2 = True
    #         else:
    #             t1_cond = [True] * dif
    #             t2_cond = [True] * dif
    #
    #             t1 = [-1] * dif + current_numbers_list
    #             infer_t1_list = infer_number(t1)  # 前补
    #             t2 = current_numbers_list + [-1] * dif
    #             infer_t2_list = infer_number(t2)  # 后补
    #
    #             for i in range(0, dif):
    #                 t1_infer = infer_t1_list[i]
    #                 t2_infer = infer_t2_list[-i - 1]
    #                 if t1_infer == 0 or t1_infer in all_nums:
    #                     t1_cond[i] = False
    #                 if t2_infer in all_nums:
    #                     t2_cond[i] = False
    #             cond1 = not (False in t1_cond)
    #             cond2 = not (False in t2_cond)
    #
    #         if cond1 and not cond2:
    #             current_loc = current_choice_m["loc"]
    #             current_height = current_loc[3] - current_loc[1]
    #
    #             infer_height = max((choice_m_mean_height - current_height), int(dif * current_height / current_len))
    #             choice_m_numbers_list[e_index]["loc"][1] = current_loc[1] - infer_height
    #             choice_m_numbers_list[e_index]["loc"][5] = (choice_m_numbers_list[e_index]["loc"][1] +
    #                                                         (choice_m_numbers_list[e_index]["loc"][3] -
    #                                                          choice_m_numbers_list[e_index]["loc"][1]) // 2)
    #             choice_m_numbers_list[e_index]["numbers"] = infer_t1_list
    #             all_nums.extend(infer_t1_list[:dif])
    #         if not cond1 and cond2:
    #             current_loc = current_choice_m["loc"]
    #             current_height = current_loc[3] - current_loc[1]
    #
    #             infer_height = max((choice_m_mean_height - current_height), int(dif * current_height / current_len))
    #             infer_bottom = min(current_loc[3] + infer_height, limit_height - 1)
    #             if infer_bottom <= limit_height:
    #                 choice_m_numbers_list[e_index]["loc"][3] = infer_bottom
    #                 choice_m_numbers_list[e_index]["loc"][5] = (choice_m_numbers_list[e_index]["loc"][1] +
    #                                                             (choice_m_numbers_list[e_index]["loc"][3] -
    #                                                              choice_m_numbers_list[e_index]["loc"][1]) // 2)
    #                 choice_m_numbers_list[e_index]["numbers"] = infer_t2_list
    #                 all_nums.extend(infer_t2_list[-dif:])
    #         else:
    #             # cond1 = cond2 = true, 因为infer选择题时已横向排序， 默认这种情况不会出现
    #             pass

    direction180, direction90 = 0, 0
    for ele in choice_m_numbers_list:
        loc = ele["loc"]
        if loc[3] - loc[1] >= 2 * (loc[2] - loc[0]):
            direction = 180
            direction180 += 1
        else:
            direction = 90
            direction90 += 1
        ele.update({'direction': direction})

    # 判断大多数choice_m的方向
    if direction180 >= direction90:   # 横排

        choice_m_numbers_list = sorted(choice_m_numbers_list, key=lambda x: x['loc'][3] - x['loc'][1], reverse=True)
        choice_m_numbers_right_limit = max([ele['loc'][2] for ele in choice_m_numbers_list])
        remain_len = len(choice_m_numbers_list)
        choice_m_list = list()
        need_revised_choice_m_list = list()
        while remain_len > 0:
            # 先确定属于同行的数据，然后找字母划分block

            random_index = 0
            # print(random_index)
            ymax_limit = choice_m_numbers_list[random_index]["loc"][3]
            ymin_limit = choice_m_numbers_list[random_index]["loc"][1]

            # 当前行的choice_m
            current_row_choice_m_d = [ele for ele in choice_m_numbers_list if ymin_limit < ele["loc"][5] < ymax_limit]
            current_row_choice_m_d = sorted(current_row_choice_m_d, key=lambda x: x["loc"][0])
            # current_row_choice_m_d.append(choice_m_numbers_list[random_index])

            # 对同行的题号区域排序， 得到分割间隔， 两个题号中间的区域为choice_m
            split_pix = sorted([ele["loc"][0] for ele in current_row_choice_m_d])  # xmin排序
            split_index = get_split_index(split_pix, dif=choice_s_width * 0.8)
            split_pix = [split_pix[ele] for ele in split_index[:-1]]

            block_list = []
            for i in range(len(split_index) - 1):
                block = current_row_choice_m_d[split_index[i]: split_index[i + 1]]
                if len(block) > 1:
                    remain_len = remain_len - (len(block) - 1)
                    numbers_new = []
                    loc_new = [[], [], [], []]
                    for blk in block:
                        loc_old = blk["loc"]
                        numbers_new.extend(blk["numbers"])
                        for ii in range(4):
                            loc_new[ii].append(loc_old[ii])

                    loc_new[0] = min(loc_new[0])
                    loc_new[1] = min(loc_new[1])
                    loc_new[2] = max(loc_new[2])
                    loc_new[3] = max(loc_new[3])

                    loc_new.append(loc_new[0] + (loc_new[2] - loc_new[0]) // 2)
                    loc_new.append(loc_new[1] + (loc_new[3] - loc_new[1]) // 2)

                    block = [{"numbers": sorted(numbers_new), "loc": loc_new, "direction": block[0]["direction"]}]

                block_list.extend(block)

            current_row_choice_m_d = block_list
            current_row_chars = [ele for ele in chars_list
                                 if ymin_limit < (ele["location"]["top"] + ele["location"]["height"] // 2) < ymax_limit]

            # split_index.append(row_chars_xmax)  # 边界
            split_pix.append(limit_right)
            for i in range(0, len(split_pix) - 1):
                left_limit = split_pix[i]
                right_limit = split_pix[i + 1]
                block_chars = [ele for ele in current_row_chars
                               if left_limit < (ele["location"]["left"] + ele["location"]["width"] // 2) < right_limit]

                a_z = '_ABCD_FGHT'
                letter_index = [a_z.index(ele['char'].upper()) for ele in block_chars if ele['char'].upper() in a_z]

                letter_index_times = {ele: 0 for ele in set(letter_index)}
                for l_index in letter_index:
                    letter_index_times[l_index] += 1

                if (a_z.index("T") in letter_index) and (a_z.index("F") in letter_index):
                    choice_option = "T, F"
                    cols = 2
                else:
                    if len(letter_index) < 1:
                        tmp = 4
                        choice_option = 'A,B,C,D'
                    else:
                        tmp = max(set(letter_index))

                        choice_option = ",".join(a_z[min(letter_index):tmp + 1])
                    cols = tmp

                bias = 3  # pix
                current_loc = current_row_choice_m_d[i]["loc"]
                location = dict(xmin=(current_loc[2] + bias),  # 当前数字xmax右边
                                # xmin=max(current_loc[2] + bias, chars_xmin) + limit_left,
                                ymin=current_loc[1],

                                xmax=(right_limit - bias),
                                # xmax=min(chars_xmax, right_limit - bias) + limit_left,
                                ymax=current_loc[3])

                try:
                    # 调整choice-m区域， 避免推断出来的区域过大
                    choice_m_img = utils.crop_region(image, location)
                    if 0 in choice_m_img.shape[:2]:
                        continue
                    right_loc, bottom_loc = adjust_choice_m(choice_m_img, mean_height, mean_width * 2)
                    if right_loc > 0:
                        location.update(dict(xmax=right_loc + location['xmin']))
                    if bottom_loc > 0:
                        location.update(dict(ymax=bottom_loc + location['ymin']))
                except Exception as e:
                    print(e)
                    traceback.print_exc()

                tmp_w, tmp_h = location['xmax'] - location['xmin'], location['ymax'] - location['ymin'],
                numbers = current_row_choice_m_d[i]["numbers"]

                direction = current_row_choice_m_d[i]["direction"]
                if direction == 180:
                    choice_m = dict(class_name='choice_m',
                                    number=numbers,
                                    bounding_box=location,
                                    choice_option=choice_option,
                                    default_points=[5] * len(numbers),
                                    direction=direction,
                                    cols=cols,
                                    rows=len(numbers))
                else:
                    choice_m = dict(class_name='choice_m',
                                    number=numbers,
                                    bounding_box=location,
                                    choice_option=choice_option,
                                    default_points=[5] * len(numbers),
                                    direction=direction,
                                    cols=len(numbers),
                                    rows=cols)

                if tmp_w > 2 * choice_s_width:
                    need_revised_choice_m_list.append(choice_m)
                else:
                    choice_m_list.append(choice_m)

            remain_len = remain_len - len(current_row_choice_m_d)
            for ele in choice_m_numbers_list.copy():
                if ele in current_row_choice_m_d:
                    choice_m_numbers_list.remove(ele)

            for ele in choice_m_numbers_list.copy():
                if ele in current_row_chars:
                    choice_m_numbers_list.remove(ele)

            # 解决单行问题
            if len(choice_m_list) > 0:
                crt_right_max = max([int(ele['bounding_box']['xmax']) for ele in choice_m_list])
                if limit_right - crt_right_max > choice_s_width:
                    # 存在区域
                    region_loc = {'xmin': crt_right_max + 10,
                                  'ymin': choice_m_list[0]['bounding_box']['ymin'],
                                  'xmax': limit_right,
                                  'ymax': choice_m_list[0]['bounding_box']['ymax']}

                    contain_dig = []
                    for i, ele in enumerate(digital_loc_arr):
                        if region_loc['xmin'] < ele[0] < region_loc['xmax'] and region_loc['ymin'] < ele[1] < region_loc['ymax']:
                            contain_dig.append(digital_list[i])

                    contain_chars = [ele for ele in chars_list
                                     if region_loc['xmin'] < (ele["location"]["left"] + ele["location"]["width"] // 2) < region_loc['xmax']
                                     and
                                     region_loc['xmin'] < (ele["location"]["top"] + ele["location"]["height"] // 2) < region_loc['ymax']]
                    numbers = [-1]
                    if contain_dig or contain_chars:
                        d_ymin, d_ymax, d_xmin, d_xmax = 9999, 0, 9999, 0
                        if contain_dig:
                            numbers = [ele["digital"] for ele in contain_dig]
                            d_ymin = min([ele['loc'][1] for ele in contain_dig])
                            d_ymax = max([ele['loc'][3] for ele in contain_dig])
                            d_xmin = min([ele['loc'][0] for ele in contain_dig])
                            d_xmax = max([ele['loc'][2] for ele in contain_dig])

                        c_ymin, c_ymax, c_xmin, c_xmax = 9999, 0, 9999, 0
                        if contain_chars:
                            c_ymin = min([ele["location"]["top"] for ele in contain_chars])
                            c_ymax = max([ele["location"]["top"] + ele["location"]["height"] for ele in contain_chars])
                            c_xmin = min([ele["location"]["left"] for ele in contain_chars])
                            c_xmax = max([ele["location"]["left"] + ele["location"]["width"] for ele in contain_chars])

                        r_ymin, r_ymax = min(d_ymin, c_ymin), max(d_ymax, c_ymax)
                        r_xmin, r_xmax = min(d_xmin, c_xmin), max(d_xmax, c_xmax)

                        region_loc['ymin'] = r_ymin - 10
                        region_loc['ymax'] = r_ymax + 10
                        if d_xmin == r_xmin:
                            region_loc['xmin'] = d_xmax + 5
                            region_loc['xmax'] = d_xmax + 5 + int(1.2 * choice_s_width)
                        else:
                            if 1.2 * (r_xmax - r_xmin) > choice_s_width:
                                region_loc['xmin'] = r_xmin - 10
                                region_loc['xmax'] = r_xmax + 10
                            else:
                                region_loc['xmin'] = max((r_xmax - r_xmin) // 2 + r_xmin - choice_s_width,
                                                         crt_right_max + 10)
                                region_loc['xmax'] = min((r_xmax - r_xmin) // 2 + r_xmin + choice_s_width ,
                                                         limit_right)

                    else:
                        # 默认这种情况只有1行或2行
                        numbers = [-1]
                        region_xmin = crt_right_max + 5
                        region_xmax = int(region_xmin + 1.2 * choice_s_width)
                        region_ymin = min([int(ele['bounding_box']['ymin']) for ele in choice_m_list])
                        region_ymax = max([int(ele['bounding_box']['ymax']) for ele in choice_m_list])
                        region_ymax = region_ymin + (region_ymax - region_ymin) // 2  # 默认这种情况只有1行或2行
                        region_loc = {'xmin': region_xmin, 'ymin': region_ymin, 'xmax': region_xmax, 'ymax': region_ymax}

                    try:
                        choice_m_img = utils.crop_region(image, region_loc)
                        if 0 in choice_m_img.shape[:2]:
                            continue
                        right_loc, bottom_loc = adjust_choice_m(choice_m_img, mean_height, mean_width * 2)
                        if right_loc > 0:
                            region_loc.update(dict(xmax=right_loc + region_loc['xmin']))
                        if bottom_loc > 0:
                            region_loc.update(dict(ymax=bottom_loc + region_loc['ymin']))
                    except Exception as e:
                        print(e)
                        traceback.print_exc()

                    choice_m = dict(class_name='choice_m',
                                    number=numbers,
                                    bounding_box=region_loc,
                                    choice_option='A,B,C,D',
                                    default_points=[5],
                                    direction=180,
                                    cols=4,
                                    rows=1,
                                    single_width=(region_loc['xmax'] - region_loc['xmin']) // 4,
                                    )
                    choice_m_list.append(choice_m)

        # 单独一行不聚类(理论上不会再到这一步了, 上个block解决)
        for i, revised_choice_m in enumerate(need_revised_choice_m_list):
            loc = revised_choice_m['bounding_box']
            left_part_loc = loc.copy()
            left_part_loc.update({'xmax': loc['xmin'] + choice_s_width})
            choice_m_img = utils.crop_region(image, left_part_loc)
            right_loc, bottom_loc = adjust_choice_m(choice_m_img, mean_height, mean_width * 2)
            if right_loc > 0:
                left_part_loc.update(dict(xmax=right_loc + left_part_loc['xmin']))
            if bottom_loc > 0:
                left_part_loc.update(dict(ymax=bottom_loc + left_part_loc['ymin']))

            left_tmp_height = left_part_loc['ymax'] - left_part_loc['ymin']

            right_part_loc = loc.copy()
            # right_part_loc.update({'xmin': loc['xmax']-choice_s_width})
            right_part_loc.update({'xmin': left_part_loc['xmax'] + 5})
            choice_m_img = utils.crop_region(image, right_part_loc)
            right_loc, bottom_loc = adjust_choice_m(choice_m_img, mean_height, mean_width * 2)
            if right_loc > 0:
                right_part_loc.update(dict(xmax=right_loc + right_part_loc['xmin']))
            if bottom_loc > 0:
                right_part_loc.update(dict(ymax=bottom_loc + right_part_loc['ymin']))

            right_tmp_height = right_part_loc['ymax'] - right_part_loc['ymin']

            number_len = max(1, int(revised_choice_m['rows'] // (left_tmp_height // right_tmp_height)))
            number = [ele + revised_choice_m['number'][-1] + 1 for ele in range(number_len)]
            rows = len(number)

            revised_choice_m.update({'bounding_box': left_part_loc})
            choice_m_list.append(revised_choice_m)

            tmp = revised_choice_m.copy()
            tmp.update({'bounding_box': right_part_loc, 'number': number, 'rows': rows})
            choice_m_list.append(tmp)

        choice_m_list_copy = choice_m_list.copy()
        for ele in choice_m_list_copy:
            loc = ele["bounding_box"]
            w, h = loc['xmax'] - loc['xmin'], loc['ymax'] - loc['ymin']
            if 2 * w * h < choice_s_width * choice_s_height:
                choice_m_list.remove(ele)
        return choice_m_list

    else:   # 竖排
        # 横向最大
        choice_m_numbers_list = sorted(choice_m_numbers_list, key=lambda x: x['loc'][2] - x['loc'][0], reverse=True)
        remain_len = len(choice_m_numbers_list)
        choice_m_list = list()
        need_revised_choice_m_list = list()
        while remain_len > 0:
            # 先确定属于同列的数据，然后找字母划分block
            random_index = 0
            xmax_limit = choice_m_numbers_list[random_index]["loc"][2]
            xmin_limit = choice_m_numbers_list[random_index]["loc"][0]
            # choice_m_numbers_list.pop(random_index)

            # 当前行的choice_m
            current_row_choice_m_d = [ele for ele in choice_m_numbers_list if xmin_limit < ele["loc"][4] < xmax_limit]
            current_row_choice_m_d = sorted(current_row_choice_m_d, key=lambda x: x["loc"][1])
            # current_row_choice_m_d.append(choice_m_numbers_list[random_index])
            split_pix = sorted([ele["loc"][1] for ele in current_row_choice_m_d])  # ymin排序
            split_index = get_split_index(split_pix, dif=choice_s_height * 0.8)
            split_pix = [split_pix[ele] for ele in split_index[:-1]]

            block_list = []
            for i in range(len(split_index) - 1):
                block = current_row_choice_m_d[split_index[i]: split_index[i + 1]]
                if len(block) > 1:
                    remain_len = remain_len - (len(block) - 1)
                    numbers_new = []
                    loc_new = [[], [], [], []]
                    for blk in block:
                        loc_old = blk["loc"]
                        numbers_new.extend(blk["numbers"])
                        for ii in range(4):
                            loc_new[ii].append(loc_old[ii])

                    loc_new[0] = min(loc_new[0])
                    loc_new[1] = min(loc_new[1])
                    loc_new[2] = max(loc_new[2])
                    loc_new[3] = max(loc_new[3])

                    loc_new.append(loc_new[0] + (loc_new[2] - loc_new[0]) // 2)
                    loc_new.append(loc_new[1] + (loc_new[3] - loc_new[1]) // 2)

                    block = [{"numbers": sorted(numbers_new), "loc": loc_new, "direction": block[0]["direction"]}]

                block_list.extend(block)

            current_row_choice_m_d = block_list
            current_row_chars = [ele for ele in chars_list
                                 if xmin_limit < (ele["location"]["top"] + ele["location"]["height"] // 2) < xmax_limit]

            split_pix.append(limit_bottom)
            for i in range(0, len(split_pix) - 1):
                top_limit = split_pix[i]
                bottom_limit = split_pix[i + 1]
                block_chars = [ele for ele in current_row_chars
                               if top_limit < (ele["location"]["left"] + ele["location"]["width"] // 2) < bottom_limit]

                a_z = '_ABCD_FGHT'
                letter_text = set([ele['char'].upper() for ele in block_chars if ele['char'].upper() in a_z])
                letter_index = [a_z.index(ele['char'].upper()) for ele in block_chars if ele['char'].upper() in a_z]

                letter_index_times = {ele: 0 for ele in set(letter_index)}
                for l_index in letter_index:
                    letter_index_times[l_index] += 1

                if (a_z.index("T") in letter_index) and (a_z.index("F") in letter_index):
                    choice_option = "T, F"
                    cols = 2
                else:
                    if len(letter_index) < 1:
                        tmp = 4
                        choice_option = 'A,B,C,D'
                    else:
                        tmp = max(set(letter_index))
                        choice_option = ",".join(a_z[min(letter_index):tmp + 1])
                    cols = tmp

                bias = 3  # pix
                current_loc = current_row_choice_m_d[i]["loc"]
                location = dict(xmin=current_loc[0],
                                ymin=current_loc[3] + bias,
                                xmax=current_loc[1],
                                ymax=bottom_limit - bias)

                try:
                    choice_m_img = utils.crop_region(image, location)
                    right_loc, bottom_loc = adjust_choice_m(choice_m_img, mean_height, mean_width * 2)
                    if right_loc > 0:
                        location.update(dict(xmax=right_loc + location['xmin']))
                    if bottom_loc > 0:
                        location.update(dict(ymax=bottom_loc + location['ymin']))
                except Exception as e:
                    print(e)
                    traceback.print_exc()

                tmp_w, tmp_h = location['xmax'] - location['xmin'], location['ymax'] - location['ymin'],
                numbers = current_row_choice_m_d[i]["numbers"]
                direction = current_row_choice_m_d[i]["direction"]
                if direction == 180:
                    choice_m = dict(class_name='choice_m',
                                    number=numbers,
                                    bounding_box=location,
                                    choice_option=choice_option,
                                    default_points=[5] * len(numbers),
                                    direction=direction,
                                    cols=cols,
                                    rows=len(numbers))
                else:
                    choice_m = dict(class_name='choice_m',
                                    number=numbers,
                                    bounding_box=location,
                                    choice_option=choice_option,
                                    default_points=[5] * len(numbers),
                                    direction=direction,
                                    cols=len(numbers),
                                    rows=cols)

                if tmp_h > 2 * choice_s_height:
                    need_revised_choice_m_list.append(choice_m)
                else:
                    choice_m_list.append(choice_m)

            remain_len = remain_len - len(current_row_choice_m_d)
            for ele in choice_m_numbers_list.copy():
                if ele in current_row_choice_m_d:
                    choice_m_numbers_list.remove(ele)

            for ele in choice_m_numbers_list.copy():
                if ele in current_row_chars:
                    choice_m_numbers_list.remove(ele)

        choice_m_list_copy = choice_m_list.copy()
        for ele in choice_m_list_copy:
            loc = ele["bounding_box"]
            w, h = loc['xmax'] - loc['xmin'], loc['ymax'] - loc['ymin']
            if 2 * w * h < choice_s_width * choice_s_height:
                choice_m_list.remove(ele)

        return choice_m_list


def infer_choice_m(image, tf_sheet, infer_box_list, col_split_x, xml=None):
    # infer_box_list = ocr2sheet(image, col_split_x, ocr, xml)
    if not infer_box_list:
        for ele in tf_sheet:
            if ele['class_name'] == 'choice':
                choice_xmin = ele['bounding_box']['xmin']
                choice_ymin = ele['bounding_box']['ymin']
                choice_xmax = ele['bounding_box']['xmax']
                choice_ymax = ele['bounding_box']['ymax']

                mid_x = choice_xmin + (choice_xmax-choice_xmin)//2
                for i in range(len(col_split_x)-1):
                    if col_split_x[i] < mid_x < col_split_x[i+1]:
                        choice_xmax = col_split_x[i+1] - 5
                        infer_box_list.append({'loc': [choice_xmin, choice_ymin, choice_xmax, choice_ymax]})
                        break

    choice_s_h_list = [int(ele['bounding_box']['ymax']) - int(ele['bounding_box']['ymin']) for ele in tf_sheet
                       if ele['class_name'] == 'choice_s']
    if choice_s_h_list:
        choice_s_height = sum(choice_s_h_list) // len(choice_s_h_list)
    else:
        choice_s_height = 0

    choice_s_w_list = [int(ele['bounding_box']['xmax']) - int(ele['bounding_box']['xmin']) for ele in tf_sheet
                       if ele['class_name'] == 'choice_s']
    if choice_s_w_list:
        choice_s_width = sum(choice_s_w_list) // len(choice_s_w_list)

    else:
        choice_s_width = 0

    choice_n_list = [ele for ele in tf_sheet if ele['class_name'] == 'choice_n']

    choice_m_list = []
    for infer_box in infer_box_list:
        # {'loc': [240, 786, 1569, 1368]}
        loc = infer_box['loc']
        xmin, ymin, xmax, ymax = loc[0], loc[1], loc[2], loc[3]
        choice_flag = False

        for ele in tf_sheet:
            if ele['class_name'] in ['choice_m', 'choice_s']:
                tf_loc = ele['bounding_box']
                tf_loc_l = tf_loc['xmin']
                tf_loc_t = tf_loc['ymin']
                if xmin < tf_loc_l < xmax and ymin < tf_loc_t < ymax:
                    choice_flag = True
                    break

        if choice_flag:
            infer_image = utils.crop_region_direct(image, loc)
            ocr = brain_api.get_ocr_text_and_coordinate11(infer_image, 'accurate', 'CHN_ENG')
            # try:
            #     save_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     save_path = os.path.join(save_dir, 'choice.jpeg')
            #     cv2.imwrite(save_path, infer_image)
            #     img_tmp = utils.read_single_img(save_path)
            #     os.remove(save_path)
            #     ocr = brain_api.get_ocr_text_and_coordinate(img_tmp, 'accurate', 'CHN_ENG')
            # except Exception as e:
            #     print('write choice and ocr failed')
            #     traceback.print_exc()
            #     ocr = brain_api.get_ocr_text_and_coordinate(infer_image, 'accurate', 'CHN_ENG')

            try:
                digital_list, chars_list, digital_mean_h, digital_mean_w = find_digital(ocr, xmin, ymin)
                if not digital_list:
                    continue

                choice_m = cluster_and_anti_abnormal(image, xml, choice_n_list, digital_list, chars_list,
                                                     digital_mean_h, digital_mean_w,
                                                     choice_s_height, choice_s_width, loc)

                choice_m_list.extend(choice_m)
            except Exception as e:
                choice_m_numbers_res = []
                traceback.print_exc()
                print('not found choice feature')
                pass

    # print(choice_m_list)
    # tf_choice_sheet = [ele for ele in tf_sheet if ele['class_name'] == 'choice_m']

    if choice_m_list:
        sheet_tmp = choice_m_list.copy()
        remove_index = []
        for i, region in enumerate(sheet_tmp):
            if i not in remove_index:
                box = region['bounding_box']
                for j, region_in in enumerate(sheet_tmp):
                    box_in = region_in['bounding_box']
                    iou = utils.cal_iou(box, box_in)
                    if iou[0] > 0.85 and i != j:
                        choice_m_list.remove(region)
                        remove_index.append(j)
                        break

    print('choice_m_list:', choice_m_list)
    return choice_m_list


if __name__ == '__main__':
    img_path = r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\1.jpg'
    image = cv2.imread(img_path)
    tf_sheet = [{'class_name': 'attention', 'bounding_box': {'xmin': 282, 'ymin': 518, 'xmax': 1133, 'ymax': 798}, 'score': '0.9984'}, {'class_name': 'bar_code', 'bounding_box': {'xmin': 1242, 'ymin': 843, 'xmax': 1627, 'ymax': 1495}, 'score': '0.9999'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 379, 'ymin': 1718, 'xmax': 706, 'ymax': 2038}, 'score': '0.9996'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 843, 'ymin': 1716, 'xmax': 1168, 'ymax': 2041}, 'score': '0.9995'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 377, 'ymin': 2282, 'xmax': 699, 'ymax': 2602}, 'score': '0.9981'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 327, 'ymin': 1714, 'xmax': 377, 'ymax': 2046}, 'score': '0.9998'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 783, 'ymin': 1721, 'xmax': 838, 'ymax': 2038}, 'score': '0.9998'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 319, 'ymin': 2269, 'xmax': 381, 'ymax': 2604}, 'score': '0.9985'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 386, 'ymin': 1969, 'xmax': 689, 'ymax': 2026}, 'score': '0.9952'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 858, 'ymin': 1813, 'xmax': 1153, 'ymax': 1865}, 'score': '0.9949'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 399, 'ymin': 2373, 'xmax': 689, 'ymax': 2425}, 'score': '0.9943'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 399, 'ymin': 1815, 'xmax': 689, 'ymax': 1867}, 'score': '0.9922'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 853, 'ymin': 1971, 'xmax': 1148, 'ymax': 2024}, 'score': '0.9920'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 396, 'ymin': 1890, 'xmax': 689, 'ymax': 1949}, 'score': '0.9900'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 855, 'ymin': 1890, 'xmax': 1155, 'ymax': 1949}, 'score': '0.9821'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 394, 'ymin': 2294, 'xmax': 684, 'ymax': 2339}, 'score': '0.9744'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 389, 'ymin': 2532, 'xmax': 689, 'ymax': 2584}, 'score': '0.9700'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 396, 'ymin': 2445, 'xmax': 684, 'ymax': 2510}, 'score': '0.9669'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 863, 'ymin': 1731, 'xmax': 1145, 'ymax': 1783}, 'score': '0.8575'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 396, 'ymin': 1731, 'xmax': 689, 'ymax': 1778}, 'score': '0.7297'}, {'class_name': 'cloze', 'bounding_box': {'xmin': 1872, 'ymin': 416, 'xmax': 3219, 'ymax': 1113}, 'score': '0.9992'}, {'class_name': 'cloze_s', 'bounding_box': {'xmin': 1828, 'ymin': 456, 'xmax': 3207, 'ymax': 620}, 'score': '0.9970'}, {'class_name': 'cloze_s', 'bounding_box': {'xmin': 1835, 'ymin': 612, 'xmax': 3192, 'ymax': 788}, 'score': '0.9959'}, {'class_name': 'cloze_s', 'bounding_box': {'xmin': 1820, 'ymin': 930, 'xmax': 3232, 'ymax': 1091}, 'score': '0.9958'}, {'class_name': 'cloze_s', 'bounding_box': {'xmin': 1830, 'ymin': 771, 'xmax': 3219, 'ymax': 937}, 'score': '0.9940'}, {'class_name': 'cloze_score', 'bounding_box': {'xmin': 2827, 'ymin': 483, 'xmax': 3202, 'ymax': 610}, 'score': '0.9985'}, {'class_name': 'cloze_score', 'bounding_box': {'xmin': 2835, 'ymin': 642, 'xmax': 3197, 'ymax': 763}, 'score': '0.9984'}, {'class_name': 'cloze_score', 'bounding_box': {'xmin': 2832, 'ymin': 791, 'xmax': 3197, 'ymax': 915}, 'score': '0.9980'}, {'class_name': 'cloze_score', 'bounding_box': {'xmin': 2825, 'ymin': 945, 'xmax': 3199, 'ymax': 1071}, 'score': '0.9979'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 649, 'ymin': 895, 'xmax': 714, 'ymax': 1493}, 'score': '0.9998'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 716, 'ymin': 897, 'xmax': 778, 'ymax': 1483}, 'score': '0.9995'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 381, 'ymin': 890, 'xmax': 448, 'ymax': 1505}, 'score': '0.9994'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 448, 'ymin': 897, 'xmax': 515, 'ymax': 1498}, 'score': '0.9993'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 580, 'ymin': 890, 'xmax': 647, 'ymax': 1495}, 'score': '0.9993'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 317, 'ymin': 890, 'xmax': 381, 'ymax': 1490}, 'score': '0.9992'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 778, 'ymin': 883, 'xmax': 843, 'ymax': 1493}, 'score': '0.9988'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 515, 'ymin': 888, 'xmax': 580, 'ymax': 1495}, 'score': '0.9965'}, {'class_name': 'exam_number_w', 'bounding_box': {'xmin': 285, 'ymin': 821, 'xmax': 885, 'ymax': 907}, 'score': '0.9980'}, {'class_name': 'executor', 'bounding_box': {'xmin': 942, 'ymin': 431, 'xmax': 1334, 'ymax': 533}, 'score': '0.9354'}, {'class_name': 'full_filling', 'bounding_box': {'xmin': 1406, 'ymin': 515, 'xmax': 1652, 'ymax': 801}, 'score': '0.9421'}, {'class_name': 'info_title', 'bounding_box': {'xmin': 419, 'ymin': 186, 'xmax': 1475, 'ymax': 456}, 'score': '0.7392'}, {'class_name': 'mark', 'bounding_box': {'xmin': 1805, 'ymin': 1508, 'xmax': 3222, 'ymax': 1599}, 'score': '0.9184'}, {'class_name': 'mark', 'bounding_box': {'xmin': 3373, 'ymin': 992, 'xmax': 4784, 'ymax': 1079}, 'score': '0.9056'}, {'class_name': 'page', 'bounding_box': {'xmin': 2457, 'ymin': 3424, 'xmax': 2702, 'ymax': 3458}, 'score': '0.9778'}, {'class_name': 'page', 'bounding_box': {'xmin': 874, 'ymin': 3421, 'xmax': 1115, 'ymax': 3455}, 'score': '0.8005'}, {'class_name': 'qr_code', 'bounding_box': {'xmin': 1145, 'ymin': 523, 'xmax': 1406, 'ymax': 806}, 'score': '0.9812'}, {'class_name': 'seal_area', 'bounding_box': {'xmin': 9, 'ymin': 111, 'xmax': 190, 'ymax': 3504}, 'score': '0.9959'}, {'class_name': 'solve', 'bounding_box': {'xmin': 1788, 'ymin': 1532, 'xmax': 3229, 'ymax': 3388}, 'score': '0.9993'}, {'class_name': 'solve', 'bounding_box': {'xmin': 3343, 'ymin': 1019, 'xmax': 4787, 'ymax': 3400}, 'score': '0.9984'}, {'class_name': 'solve', 'bounding_box': {'xmin': 3356, 'ymin': 203, 'xmax': 4782, 'ymax': 974}, 'score': '0.9073'}, {'class_name': 'student_info', 'bounding_box': {'xmin': 4, 'ymin': 2406, 'xmax': 133, 'ymax': 3068}, 'score': '0.9966'}, {'class_name': 'time', 'bounding_box': {'xmin': 166, 'ymin': 421, 'xmax': 587, 'ymax': 518}, 'score': '0.9802'}, {'class_name': 'total_score', 'bounding_box': {'xmin': 548, 'ymin': 431, 'xmax': 942, 'ymax': 520}, 'score': '0.9970'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 152, 'ymin': 1544, 'xmax': 1796, 'ymax': 1587}, 'score': '0.9988'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1839, 'ymin': 1601, 'xmax': 2427, 'ymax': 1648}, 'score': '0.9983'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 150, 'ymin': 2169, 'xmax': 1472, 'ymax': 2212}, 'score': '0.9978'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 3404, 'ymin': 1079, 'xmax': 3988, 'ymax': 1126}, 'score': '0.9948'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1772, 'ymin': 1338, 'xmax': 3279, 'ymax': 1383}, 'score': '0.9911'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1776, 'ymin': 276, 'xmax': 3289, 'ymax': 321}, 'score': '0.9850'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1850, 'ymin': 493, 'xmax': 1952, 'ymax': 597}, 'score': '0.9964'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 183, 'ymin': 2073, 'xmax': 292, 'ymax': 2165}, 'score': '0.9958'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1845, 'ymin': 786, 'xmax': 1954, 'ymax': 917}, 'score': '0.9957'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1847, 'ymin': 635, 'xmax': 1952, 'ymax': 763}, 'score': '0.9957'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1847, 'ymin': 945, 'xmax': 1952, 'ymax': 1074}, 'score': '0.9956'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 193, 'ymin': 1508, 'xmax': 295, 'ymax': 1604}, 'score': '0.9946'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1748, 'ymin': 1312, 'xmax': 1852, 'ymax': 1389}, 'score': '0.9932'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1748, 'ymin': 235, 'xmax': 1857, 'ymax': 329}, 'score': '0.9901'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1838, 'ymin': 1590, 'xmax': 1927, 'ymax': 1669}, 'score': '0.8317'}, {'class_name': 'verify', 'bounding_box': {'xmin': 1312, 'ymin': 416, 'xmax': 1686, 'ymax': 515}, 'score': '0.8929'}, {'class_name': 'bar_code', 'bounding_box': {'xmin': 1305, 'ymin': 844, 'xmax': 1634, 'ymax': 1516}}, {'class_name': 'exam_number', 'bounding_box': {'xmin': 285, 'ymin': 883, 'xmax': 885, 'ymax': 1505}}]
    infer_box_list = [{'loc': [48, 893, 1768, 1498]}, {'loc': [48, 1802, 1768, 2087]}, {'loc': [48, 2279, 1768, 2596]}, {'loc': [1768, 532, 3281, 1043]}]
    col_split_x = [1768, 3281]
    infer_choice_m(image, tf_sheet, infer_box_list, col_split_x)