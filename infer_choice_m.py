# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/22 10:27'


import numpy as np
import re, cv2, utils, sys
from brain_api_charge import get_ocr_text_and_coordinate11
import xml.etree.ElementTree as ET
from utils import create_xml
import traceback
from sklearn.cluster import DBSCAN
from itertools import chain


# 合并字符的原则是根据单个字符合并的
def combine_char_baidu_format(words_result, left_boundary, top_boundary):
    # 只是一个block送进来ocr很容易发生words都在同一行的情况。  --> 'words': ' 4 [A] [B] [C] [D] 8 [A] [B] [C] [D]'
    all_new_char_list = []
    for i, words_line in enumerate(words_result):
        char_list = words_line['chars']
        char_str = ''
        char_box = []
        for ele in char_list:
            char = ele['char']
            loc = ele['location']
            left = loc['left'] + left_boundary  # 将坐标映射到原图上，拿到的坐标是在原图基础上的结果
            top = loc['top'] + top_boundary
            right = left + loc['width']
            bottom = top + loc['height']
            width = loc['width']
            height = loc['height']
            box = (left, top, right, bottom, width, height)
            char_box.append(box)
            char_str = char_str + char
        print(char_box)
        split_index = []
        for ii in range(1, len(char_box)):
            pre_one = char_box[ii]
            rear_one = char_box[ii - 1]
            x_dif = pre_one[0] - rear_one[2]
            y_dif = pre_one[1] - rear_one[1]
            if y_dif < rear_one[5]:
                if x_dif < rear_one[4]:
                    pass
                else:
                    split_index.append(ii)
            else:
                split_index.append(ii)
        split_index.append(len(char_box))
        split_index.insert(0, 0)

        combine_char_list = []
        # 这里合并字符串之后了
        for k in range(1, len(split_index)):
            new_char = char_str[split_index[k-1]: split_index[k]]
            new_char_box_list = char_box[split_index[k - 1]: split_index[k]]   # 这个list里是能合并的多个字符的list，每个box对应一个字符
            new_char_box_arr = np.array(new_char_box_list)

            # v1
            box_xmin = min(new_char_box_arr[:, 0])
            box_ymin = min(new_char_box_arr[:, 1])
            box_xmax = max(new_char_box_arr[:, 2])
            box_ymax = max(new_char_box_arr[:, 3])
            box_width = box_xmax - box_xmin
            box_height = box_ymax - box_ymin

            # v2
            # min_array = new_char_box_arr.min(axis=0)
            # max_array = new_char_box_arr.max(axis=0)
            # box_xmin1 = min_array[0]
            # box_ymin1 = min_array[1]
            # box_width1 = max_array[2] - min_array[0]
            # box_height1 = max_array[3] - min_array[1]
            # print(box_height1)

            location = {'left': box_xmin, 'top': box_ymin, 'width': box_width, 'height': box_height}

            single_char_box = []
            # 给出单个字的字符和坐标，给字符串和坐标
            for jj, bbox in enumerate(new_char_box_list):
                n_char = new_char[jj]
                single_char_box.append({'char': n_char, 'location': {'left': bbox[0], 'top': bbox[1], 'width': bbox[4], 'height': bbox[5]}})
            combine_char_list.append({'chars': single_char_box, 'location': location, 'words': new_char})
        all_new_char_list = all_new_char_list + combine_char_list
    # xml_template = 'exam_info/000000-template.xml'
    # tree = ET.parse(xml_template)
    # for index, ele1 in enumerate(all_new_char_list):
    #     for ele in ele1['chars']:
    #         create_xml(ele['char'], tree, ele['location']['left'], ele['location']['top'],
    #                    ele['location']['left'] + ele['location']['width'],
    #                    ele['location']['top'] + ele['location']['height'])
    # tree.write(r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\4.xml')
    return all_new_char_list


# 之前写的合并字符有问题，重新根据索引值写一稿
def find_digital(words_result, left_boundary, top_boundary):
    # 合并字符，将靠近的字符合并
    new_words_result = combine_char_baidu_format(words_result, left_boundary, top_boundary)
    pattern = '\d+'
    chars_list = []
    digital_list = []
    height_list = []
    width_list = []
    x_list = []
    y_list = []

    digital_index = []
    for words_line in new_words_result:
        words = words_line['words'].replace(' ', '')
        words_m = re.finditer(pattern, words)
        words_m_list = [(m.group(), m.span()) for m in words_m if m]
        char_index = [i for i in range(len(words_line['chars']))]
        for char in words_m_list:
            number = int(char[0])
            start_index = char[1][0]
            end_index = char[1][1] - 1   # 右边取不到，所以要去掉1
            char_start = words_line['chars'][start_index]
            char_end = words_line['chars'][end_index]
            if start_index == end_index:
                digital_index = digital_index + [start_index]
            else:
                digital_index = digital_index + char_index[start_index: end_index]
            digital_xmin = int(char_start['location']['left'])
            digital_ymin = min(int(char_start['location']['top']), (int(char_end['location']['top'])))
            digital_xmax = int(char_end['location']['left']) + int(char_end['location']['width'])
            digital_ymax = max(char_start['location']['top'] + char_start['location']['height'],
                               char_end['location']['top'] + char_end['location']['height'])
            xmid = digital_xmin + (digital_xmax - digital_xmin) // 2
            ymid = digital_ymin + (digital_ymax - digital_ymin) // 2

            x_list.append(xmid)
            y_list.append(ymid)

            height_list.append(digital_ymax - digital_ymin)
            width_list.append(digital_xmax - digital_xmin)

            number_loc = (digital_xmin, digital_ymin, digital_xmax, digital_ymax, xmid, ymid)
            digital_list.append({'number': number, 'loc': number_loc})
        current_chars = [char for index, char in enumerate(words_line['chars'])
                         if index not in digital_index and char['char'].encode('utf-8').isalpha()]
        # string.isalpha的用法: 如果字符串至少有一个字符并且所有字符都是字母则返回 True，否则返回 False
        chars_list = chars_list + current_chars   # char_list里面都是字符，没有括号，数字等，只有A,B,C,D之类的字符
    if not height_list or not width_list:
        d_mean_height, d_mean_width = 0, 0
    else:
        d_mean_height = int(np.mean(np.array(height_list)))
        d_mean_width = int(np.mean(np.array(width_list)))

    return digital_list, chars_list, d_mean_height, d_mean_width


# polygon 多边形
def point_in_polygon(point, polygon):
    xmin, ymin, xmax, ymax = polygon['xmin'], polygon['ymin'], polygon['xmax'], polygon['ymax']
    if xmin < point[0] < xmax and ymin < point[1] < ymax:
        return True
    else:
        return False


def get_split_index(list1):
    list1_array = np.array(list1)
    interval_list = list(list1_array[1:] - list1_array[:-1])

    dif = 0
    split_index = []
    for i, interval in enumerate(interval_list):
        if dif:
            split_dif = dif
        else:
            split_dif = np.mean(np.array(interval_list))
        if interval > split_dif:
            split_index.append(i + 1)
    split_index.insert(0, 0)
    split_index.append(len(list1))
    return split_index


def infer_number(number_list, times, interval):
    # 默认的interval=1，题号为公差为1的等差数列
    # 这里的times是控制递归的深度
    if times > 30:
        return number_list
    new_number_list = []
    if -1 not in number_list:
        return number_list
    elif number_list[-1] != -1:
        for i in range(1, len(number_list)):
            number = number_list[-1] - i
            new_number_list.append(number)
        new_number_list.append(number_list[-1])   # 最后一个肯定是不为-1的
        new_number_list = sorted(new_number_list)
        return new_number_list
    else:
        for n_index in range(len(number_list) - 1):
            if n_index == 0:
                if number_list[n_index] != -1:
                    number_list[n_index + 1] = number_list[n_index] + interval
            elif n_index != 0 and len(number_list) > 1 and number_list[n_index] != -1:
                if number_list[n_index - 1] == -1:
                    number_list[n_index - 1] = number_list[n_index] - interval
                elif number_list[n_index + 1] == -1:
                    number_list[n_index + 1] = number_list[n_index] + interval
        times += 1
        return infer_number(number_list, times, interval)


# 题号的补全
def cluster2choice_m(cluster, mean_width, tf_box=True):
    # 比较x的坐标，去掉误差值
    # 这里的cluster是题号的列表，这里的题号是合并过的，并且是正确的题号，但是这里的题号也有错误的，ocr问题产生的错误
    # cluster --> [left, top, right, bottom, width, height]
    number_x = [ele['loc'][4] for ele in cluster]
    number_interval = abs(np.array(number_x)[1:] - np.array(number_x)[:-1])
    number_error_index_suspect = np.where(np.array(number_interval) > mean_width)[0]
    number_error_index_interval = number_error_index_suspect[1:] - number_error_index_suspect[:-1]

    true_index = list(np.where(number_error_index_interval > 1)[0] + 1)
    # np.where()的值为空则不能进行加减操作，不为空则整个数组加1
    true_index.insert(0, 0)
    true_index.append(len(number_error_index_suspect))
    error_index = []
    for i in range(len(number_error_index_suspect) - 1):
        a = true_index[i]
        b = true_index[i + 1]
        block = list(number_error_index_suspect[a: b])
        error_index = error_index + block[1:]
    cluster = [ele for i, ele in enumerate(cluster) if i not in error_index]   # 踢掉error_index所在的索引值
    numbers = [ele['number'] for ele in cluster]
    # 到这边是考虑了x轴的横坐标，根据坐标拿到有问题的索引值，并剔除掉,截止到这边，题号发生了错误是不做修改的,eg: [2, 33, 4]这样的情况也是存在的
    # 最后一个位置的x坐标出错是不能找到对应的索引值的
    # 以此类推，第一个位置出错也是不能找到出错的索引值的
    # 只能找到中间位置出错的索引值，并加以剔除


    # 确定题号的位置，同一个block里的是一个等差数列的子集
    number_array = np.array(numbers)
    number_sum = np.array(numbers) + np.flipud(np.array(numbers))
    number_count = np.bincount(number_sum)
    number_mode_times = np.max(number_count)
    number_value = np.argmax(number_count)

    if len(numbers) != number_mode_times and number_mode_times >= 2:
        print('启动题号补全')
        numbers_arr = abs(np.array(numbers)[1:] - np.array(numbers)[:-1])
        number_count1 = np.bincount(numbers_arr)
        number_value_interval = np.argmax(number_count1)
        suspect_index = np.where(number_sum != number_value)[0]    # 这不找到suspect index了嘛，上面写那么多干啥用的
        for suspect in suspect_index:
            if suspect == 0:
                cond_left = 0   # 索引为0的时候，左边的坐标不找了
                cond_right = number_array[suspect + 1] == number_array[suspect] + number_value_interval
            elif suspect == len(numbers) - 1:
                cond_left = number_array[suspect - 1] == number_array[suspect] - number_value_interval
                cond_right = 0
            else:
                cond_left = number_array[suspect - 1] == number_array[suspect] - number_value_interval
                cond_right = number_array[suspect + 1] == number_array[suspect] + number_value_interval

            # 以下是将错的纠正但是不补数字
            if cond_left or cond_right:
                pass
            else:
                number_array[suspect] = -1   # 这边将错的那个位置变成-1，还不涉及题号的补全
        times = 0
        number_array = infer_number(number_array, times, number_value_interval)
        number_array = np.array(number_array)

        # TODO 这里的number_array仅仅是测试用
    number_interval = number_array[1:] - number_array[:-1]
    split_index = []
    for ii, split in enumerate(number_interval):
        if split > np.mean(np.array(number_interval)):
            split_index.append(ii + 1)
    split_index.append(len(number_array))
    split_index = sorted(list(set(split_index)))
    block_list = []
    if tf_box:
        split_index = [0, len(cluster)]
    for k in range(len(split_index) - 1):
        block = cluster[split_index[k]: split_index[k + 1]]
        block_number = list(number_array[split_index[k]: split_index[k + 1]])
        xmin = min([ele['loc'][0] for ele in block])
        ymin = min([ele['loc'][1] for ele in block])
        xmax = max([ele['loc'][2] for ele in block])
        ymax = max([ele['loc'][3] for ele in block])
        xmid = xmin + (xmax - xmin) // 2
        ymid = ymin + (ymax - ymin) // 2
        block_list.append({'numbers': block_number, 'loc': [xmin, ymin, xmax, ymax, xmid, ymid]})
    return block_list


# 这个函数的目的主要是聚类choice_m区域并踢掉多余的
def cluster_and_anti_abnormal(image, choice_n_list, digital_list, chars_list,
                              mean_height, mean_width, choice_s_height, choice_s_width, limit_loc):
    # TODO 仅仅为了测试使用
    # for ele in digital_list:
    #     if ele['number'] == 6:
    #         loc = list(ele['loc'])
    #         loc[-2] = 780
    #         ele.update({'loc': tuple(loc)})
    # print(digital_list)

    # 去掉一个choice_n
    choice_n_list = [{'class_name': 'choice_n', 'bounding_box': {'xmin': 783, 'ymin': 1721, 'xmax': 838, 'ymax': 2038}, 'score': '0.9998'},
                     {'class_name': 'choice_n', 'bounding_box': {'xmin': 319, 'ymin': 2269, 'xmax': 381, 'ymax': 2604}, 'score': '0.9985'}]

    limit_left, limit_top, limit_right, limit_bottom = limit_loc
    x_mid, y_mid = limit_left + (limit_right - limit_left) // 2, limit_top + (limit_bottom - limit_right) // 2
    digital_loc_arr = []
    digital_list_to_cluster = []
    # 这个for循环做的事情就是将数字放到每个choice_n中去
    # 要是找不到对应的choice_n就单独放到一个list中，接下来去聚类，期望通过聚类拿到choice_m
    for i, ele in enumerate(digital_list):
        point = [ele['loc'][-2], ele['loc'][-1]]
        contain = False
        for choice_n in choice_n_list:
            numbers = choice_n.get('numbers')
            if not numbers:
                choice_n.update({'numbers': []})
            if point_in_polygon(point, choice_n['bounding_box']) == True:
                contain = True
                choice_n['numbers'].append(digital_list[i])
                break

        if not contain:
            digital_loc_arr.append(point)
            digital_list_to_cluster.append(digital_list[i])
    choice_m_numbers_list = []
    for ele in choice_n_list:
        xmin = ele['bounding_box']['xmin']
        ymin = ele['bounding_box']['ymin']
        xmax = ele['bounding_box']['xmax']
        ymax = ele['bounding_box']['ymax']
        xmid = xmin + (xmax - xmin) // 2
        ymid = ymin + (ymax - ymin) // 2

        cluster = ele.get('numbers')
        if not cluster:
            block_list = [{'numbers': [-1], 'loc': [xmin, ymin, xmax, ymax, xmid, ymid]}]   # 题号起码有一个，所以前面用的-1
        else:
            block_list = cluster2choice_m(cluster, mean_width, tf_box=True)
            block_list[0].update({"loc": [xmin, ymin, xmax, ymax, xmid, ymid]})
            # 把choice_n的坐标更新到自己infer出来的坐标中
            # 理论上自己infer出来的坐标是对的，但是是包含字的坐标，模型出来的框是要比字的框要大的
        choice_m_numbers_list += block_list
    print(choice_m_numbers_list)
    # 下面根据数字聚类出choice_m区域
    if digital_loc_arr:
        digital_loc_arr = np.array(digital_loc_arr)
        # 一般而言,choice_s的高要比数字的均值要大
        if choice_s_height != 0:
            eps = int(choice_s_height * 2.5)
        else:
            eps = int(mean_height * 3)
        print('eps:', eps)
        db = DBSCAN(eps=eps, min_samples=2, metric='chebyshev').fit(digital_loc_arr)
        # 默认的metric是欧式距离，chebyshev是chebyshev
        # 密度聚类之DBSCAN算法需要两个参数：ε(eps) ε-邻域(这个参数是设定的半径r)和形成高密度区域所需要的最少点数(minPts)
        # https://zhuanlan.zhihu.com/p/54833132         这个链接讲的原理知识
        # https://www.cntofu.com/book/85/ml/cluster/scikit-learn-dbscan.md    官方文档
        labels = db.labels_
        # TODO这个地方的聚类主要不是为了处理离群点，而是处理数字寻找到了，但是choice_n丢了的情况，这种情况下靠聚类是能补到choice_n的区域
        # labels_：所有点的分类结果。无论核心点还是边界点，只要是同一个簇的都被赋予同样的label，噪声点为-1.
        # core_sample_indices_：核心点的索引，因为labels_不能区分核心点还是边界点，所以需要用这个索引确定核心点
        cluster_labels = []
        for ele in labels:
            if ele not in cluster_labels and ele != -1:
                cluster_labels.append(ele)
        a_e_dict = {k: [] for k in cluster_labels}
        for i, ele in enumerate(labels):
            if ele != -1:
                a_e_dict[ele].append(digital_list_to_cluster[i])
        for ele in cluster_labels:
            cluster = a_e_dict[ele]
            choice_m_numbers_list = choice_m_numbers_list + cluster2choice_m(cluster, mean_width)   # 这部分送进去聚类是想找到一个choice_n中的数字
    all_nums_list = [ele['numbers'] for ele in choice_m_numbers_list]
    all_nums_len = [len(ele) for ele in all_nums_list]
    all_nums = list(chain.from_iterable(all_nums_list))

    # 根据数字的坐标判断横排,列排
    # 这里的横排：180(高小于宽的情况)   列排：90(高大于宽的情况)

    direction180, direction90 = 0, 0
    for ele in choice_m_numbers_list:
        loc = ele['loc']
        if loc[5] >= 2 * loc[4]:
            direction = 180
            direction180 += 1
        else:
            direction = 90
            direction90 += 1
        ele.update({'direction': direction})
    # 判断大多数choice_m方向
    if direction180 > direction90:      # 字在框的左边为平行，180度；字在框的上面为垂直，90度
        choice_m_numbers_list = sorted(choice_m_numbers_list, key=lambda k: k['loc'][3] - k['loc'][1], reverse=True)
        # print(choice_m_numbers_list)
        choice_m_list = []
        need_revised_choice_m_list = []
        remian_len = len(choice_m_numbers_list)
        # loc --> [xmin, ymin, xmax, ymax, xmid, ymid]
        while remian_len > 0:
            # 先确定同行的数据，再找字母划分block
            random_index = 0
            ymax_limit = choice_m_numbers_list[random_index]['loc'][3]
            ymin_limit = choice_m_numbers_list[random_index]['loc'][1]
            # print(ymax_limit)

            current_choice_m_row = [ele for ele in choice_m_numbers_list if ymin_limit < ele['loc'][5] < ymax_limit]
            current_choice_m_row = sorted(current_choice_m_row, key=lambda k: k['loc'][0])
            # print(current_choice_m_row)

            # 对同行的题号区域进行排序，两个题号之间的区域为choice_m
            split_pix = sorted([ele['loc'][0] for ele in current_choice_m_row])   # xmin
            split_index = get_split_index(split_pix)
            # split_index[:-1]这种写法就是不包含最后一个元数,取除了最后一个元素的其他
            split_pix = [split_pix[ele] for ele in split_index[:-1]]
            block_list = []
            for i in range(len(split_index) - 1):
                block = choice_m_numbers_list[split_index[i]: split_index[i + 1]]
                if len(block) > 1:
                    remian_len = remian_len - (len(block) - 1)
                    print(remian_len)
                else:
                    block_list.append(block)
            print(block_list)


def infer_choice_m_demo(image, tf_sheet, col_split_x, infer_box_list):
    # infer_box_list里的内容就是分类出来的每个区域内容，包括任何可能的区域，choice, exam_number, cloze等
    # 首先考虑这个区域不存在的情况，自己并没有成功分类出来这个区域
    if not infer_box_list:
        # 在每栏内合并choice区域
        for ele in tf_sheet:
            if ele['class_name'] == 'choice':
                xmin = ele['bounding_box']['xmin']
                ymin = ele['bounding_box']['ymin']
                xmax = ele['bounding_box']['xmax']
                ymax = ele['bounding_box']['ymax']
                xmid = xmin + (xmax - xmin) // 2
                for i in range(len(col_split_x) - 1):
                    if col_split_x[i] < xmid < col_split_x[i + 1]:
                        choice_xmax = col_split_x[i + 1] - 5
                        infer_box_list.append({'loc': [xmin, ymin, choice_xmax, ymax]})
                        break
    # print(infer_box_list)

    # choice_s_height_list
    choice_s_h_list = [ele['bounding_box']['ymax'] - ele['bounding_box']['ymin'] for ele in tf_sheet if ele['class_name'] == 'choice_s']
    if choice_s_h_list:
        choice_s_h = sum(choice_s_h_list) // len(choice_s_h_list)
    else:
        choice_s_h = 0

    choice_s_w_list = [ele['bounding_box']['xmax'] - ele['bounding_box']['xmin'] for ele in tf_sheet if ele['class_name'] == 'choice_s']
    if choice_s_w_list:
        choice_s_w = sum(choice_s_w_list) // len(choice_s_w_list)
    else:
        choice_s_w = 0

    choice_n_list = [ele for ele in tf_sheet if ele['class_name'] == 'choice_n']

    choice_m_list = []
    for index, infer_box in enumerate(infer_box_list):
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
            choice_region = utils.crop_region_direct(image, loc)
            cv2.imwrite(r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\\' + f'choice{index}.jpg', choice_region)
            words_result = get_ocr_text_and_coordinate11(choice_region)
            # xml_template = 'exam_info/000000-template.xml'
            # tree = ET.parse(xml_template)
            # for i, ele in enumerate(words_result):
            #     create_xml(ele['words'], tree, ele['location']['left'], ele['location']['top'],
            #                ele['location']['left'] + ele['location']['width'], ele['location']['top'] + ele['location']['height'])
            # tree.write(r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\\' + f'choice{index}.xml')
            # print(words_result)
            try:
                digital_list, chars_list, digital_mean_h, digital_mean_w = find_digital(words_result, xmin, ymin)
                if not digital_list:
                    continue
                choice_m = cluster_and_anti_abnormal(image, choice_n_list, digital_list, chars_list,
                                                     digital_mean_h, digital_mean_w, choice_s_h, choice_s_w, loc)

                choice_m_list.extend(choice_m)
            except Exception as e:
                traceback.extract_tb(sys.exc_info()[2])   # 打印出具体出错的行并不打断程序
                print('not find choice region')
                pass
    print(choice_s_w)


if __name__ == '__main__':
    # img_path = r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\1.jpg'
    # image = cv2.imread(img_path)
    # tf_sheet = [{'class_name': 'attention', 'bounding_box': {'xmin': 282, 'ymin': 518, 'xmax': 1133, 'ymax': 798}, 'score': '0.9984'}, {'class_name': 'bar_code', 'bounding_box': {'xmin': 1242, 'ymin': 843, 'xmax': 1627, 'ymax': 1495}, 'score': '0.9999'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 379, 'ymin': 1718, 'xmax': 706, 'ymax': 2038}, 'score': '0.9996'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 843, 'ymin': 1716, 'xmax': 1168, 'ymax': 2041}, 'score': '0.9995'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 377, 'ymin': 2282, 'xmax': 699, 'ymax': 2602}, 'score': '0.9981'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 327, 'ymin': 1714, 'xmax': 377, 'ymax': 2046}, 'score': '0.9998'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 783, 'ymin': 1721, 'xmax': 838, 'ymax': 2038}, 'score': '0.9998'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 319, 'ymin': 2269, 'xmax': 381, 'ymax': 2604}, 'score': '0.9985'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 386, 'ymin': 1969, 'xmax': 689, 'ymax': 2026}, 'score': '0.9952'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 858, 'ymin': 1813, 'xmax': 1153, 'ymax': 1865}, 'score': '0.9949'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 399, 'ymin': 2373, 'xmax': 689, 'ymax': 2425}, 'score': '0.9943'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 399, 'ymin': 1815, 'xmax': 689, 'ymax': 1867}, 'score': '0.9922'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 853, 'ymin': 1971, 'xmax': 1148, 'ymax': 2024}, 'score': '0.9920'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 396, 'ymin': 1890, 'xmax': 689, 'ymax': 1949}, 'score': '0.9900'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 855, 'ymin': 1890, 'xmax': 1155, 'ymax': 1949}, 'score': '0.9821'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 394, 'ymin': 2294, 'xmax': 684, 'ymax': 2339}, 'score': '0.9744'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 389, 'ymin': 2532, 'xmax': 689, 'ymax': 2584}, 'score': '0.9700'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 396, 'ymin': 2445, 'xmax': 684, 'ymax': 2510}, 'score': '0.9669'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 863, 'ymin': 1731, 'xmax': 1145, 'ymax': 1783}, 'score': '0.8575'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 396, 'ymin': 1731, 'xmax': 689, 'ymax': 1778}, 'score': '0.7297'}, {'class_name': 'cloze', 'bounding_box': {'xmin': 1872, 'ymin': 416, 'xmax': 3219, 'ymax': 1113}, 'score': '0.9992'}, {'class_name': 'cloze_s', 'bounding_box': {'xmin': 1828, 'ymin': 456, 'xmax': 3207, 'ymax': 620}, 'score': '0.9970'}, {'class_name': 'cloze_s', 'bounding_box': {'xmin': 1835, 'ymin': 612, 'xmax': 3192, 'ymax': 788}, 'score': '0.9959'}, {'class_name': 'cloze_s', 'bounding_box': {'xmin': 1820, 'ymin': 930, 'xmax': 3232, 'ymax': 1091}, 'score': '0.9958'}, {'class_name': 'cloze_s', 'bounding_box': {'xmin': 1830, 'ymin': 771, 'xmax': 3219, 'ymax': 937}, 'score': '0.9940'}, {'class_name': 'cloze_score', 'bounding_box': {'xmin': 2827, 'ymin': 483, 'xmax': 3202, 'ymax': 610}, 'score': '0.9985'}, {'class_name': 'cloze_score', 'bounding_box': {'xmin': 2835, 'ymin': 642, 'xmax': 3197, 'ymax': 763}, 'score': '0.9984'}, {'class_name': 'cloze_score', 'bounding_box': {'xmin': 2832, 'ymin': 791, 'xmax': 3197, 'ymax': 915}, 'score': '0.9980'}, {'class_name': 'cloze_score', 'bounding_box': {'xmin': 2825, 'ymin': 945, 'xmax': 3199, 'ymax': 1071}, 'score': '0.9979'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 649, 'ymin': 895, 'xmax': 714, 'ymax': 1493}, 'score': '0.9998'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 716, 'ymin': 897, 'xmax': 778, 'ymax': 1483}, 'score': '0.9995'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 381, 'ymin': 890, 'xmax': 448, 'ymax': 1505}, 'score': '0.9994'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 448, 'ymin': 897, 'xmax': 515, 'ymax': 1498}, 'score': '0.9993'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 580, 'ymin': 890, 'xmax': 647, 'ymax': 1495}, 'score': '0.9993'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 317, 'ymin': 890, 'xmax': 381, 'ymax': 1490}, 'score': '0.9992'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 778, 'ymin': 883, 'xmax': 843, 'ymax': 1493}, 'score': '0.9988'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 515, 'ymin': 888, 'xmax': 580, 'ymax': 1495}, 'score': '0.9965'}, {'class_name': 'exam_number_w', 'bounding_box': {'xmin': 285, 'ymin': 821, 'xmax': 885, 'ymax': 907}, 'score': '0.9980'}, {'class_name': 'executor', 'bounding_box': {'xmin': 942, 'ymin': 431, 'xmax': 1334, 'ymax': 533}, 'score': '0.9354'}, {'class_name': 'full_filling', 'bounding_box': {'xmin': 1406, 'ymin': 515, 'xmax': 1652, 'ymax': 801}, 'score': '0.9421'}, {'class_name': 'info_title', 'bounding_box': {'xmin': 419, 'ymin': 186, 'xmax': 1475, 'ymax': 456}, 'score': '0.7392'}, {'class_name': 'mark', 'bounding_box': {'xmin': 1805, 'ymin': 1508, 'xmax': 3222, 'ymax': 1599}, 'score': '0.9184'}, {'class_name': 'mark', 'bounding_box': {'xmin': 3373, 'ymin': 992, 'xmax': 4784, 'ymax': 1079}, 'score': '0.9056'}, {'class_name': 'page', 'bounding_box': {'xmin': 2457, 'ymin': 3424, 'xmax': 2702, 'ymax': 3458}, 'score': '0.9778'}, {'class_name': 'page', 'bounding_box': {'xmin': 874, 'ymin': 3421, 'xmax': 1115, 'ymax': 3455}, 'score': '0.8005'}, {'class_name': 'qr_code', 'bounding_box': {'xmin': 1145, 'ymin': 523, 'xmax': 1406, 'ymax': 806}, 'score': '0.9812'}, {'class_name': 'seal_area', 'bounding_box': {'xmin': 9, 'ymin': 111, 'xmax': 190, 'ymax': 3504}, 'score': '0.9959'}, {'class_name': 'solve', 'bounding_box': {'xmin': 1788, 'ymin': 1532, 'xmax': 3229, 'ymax': 3388}, 'score': '0.9993'}, {'class_name': 'solve', 'bounding_box': {'xmin': 3343, 'ymin': 1019, 'xmax': 4787, 'ymax': 3400}, 'score': '0.9984'}, {'class_name': 'solve', 'bounding_box': {'xmin': 3356, 'ymin': 203, 'xmax': 4782, 'ymax': 974}, 'score': '0.9073'}, {'class_name': 'student_info', 'bounding_box': {'xmin': 4, 'ymin': 2406, 'xmax': 133, 'ymax': 3068}, 'score': '0.9966'}, {'class_name': 'time', 'bounding_box': {'xmin': 166, 'ymin': 421, 'xmax': 587, 'ymax': 518}, 'score': '0.9802'}, {'class_name': 'total_score', 'bounding_box': {'xmin': 548, 'ymin': 431, 'xmax': 942, 'ymax': 520}, 'score': '0.9970'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 152, 'ymin': 1544, 'xmax': 1796, 'ymax': 1587}, 'score': '0.9988'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1839, 'ymin': 1601, 'xmax': 2427, 'ymax': 1648}, 'score': '0.9983'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 150, 'ymin': 2169, 'xmax': 1472, 'ymax': 2212}, 'score': '0.9978'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 3404, 'ymin': 1079, 'xmax': 3988, 'ymax': 1126}, 'score': '0.9948'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1772, 'ymin': 1338, 'xmax': 3279, 'ymax': 1383}, 'score': '0.9911'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 1776, 'ymin': 276, 'xmax': 3289, 'ymax': 321}, 'score': '0.9850'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1850, 'ymin': 493, 'xmax': 1952, 'ymax': 597}, 'score': '0.9964'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 183, 'ymin': 2073, 'xmax': 292, 'ymax': 2165}, 'score': '0.9958'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1845, 'ymin': 786, 'xmax': 1954, 'ymax': 917}, 'score': '0.9957'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1847, 'ymin': 635, 'xmax': 1952, 'ymax': 763}, 'score': '0.9957'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1847, 'ymin': 945, 'xmax': 1952, 'ymax': 1074}, 'score': '0.9956'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 193, 'ymin': 1508, 'xmax': 295, 'ymax': 1604}, 'score': '0.9946'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1748, 'ymin': 1312, 'xmax': 1852, 'ymax': 1389}, 'score': '0.9932'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1748, 'ymin': 235, 'xmax': 1857, 'ymax': 329}, 'score': '0.9901'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 1838, 'ymin': 1590, 'xmax': 1927, 'ymax': 1669}, 'score': '0.8317'}, {'class_name': 'verify', 'bounding_box': {'xmin': 1312, 'ymin': 416, 'xmax': 1686, 'ymax': 515}, 'score': '0.8929'}, {'class_name': 'bar_code', 'bounding_box': {'xmin': 1305, 'ymin': 844, 'xmax': 1634, 'ymax': 1516}}, {'class_name': 'exam_number', 'bounding_box': {'xmin': 285, 'ymin': 883, 'xmax': 885, 'ymax': 1505}}]
    # infer_box_list = [{'loc': [48, 893, 1768, 1498]}, {'loc': [48, 1802, 1768, 2087]}, {'loc': [48, 2279, 1768, 2596]}, {'loc': [1768, 532, 3281, 1043]}]
    # col_split_x = [1768, 3281]
    # infer_choice_m_demo(image, tf_sheet, col_split_x, infer_box_list)


    # cluster = [{'number': 2, 'loc': (337, 1812, 363, 1862, 350, 1837)},
    #            {'number': 3, 'loc': (340, 1898, 360, 1941, 350, 1919)},
    #            {'number': 5, 'loc': (342, 1981, 363, 2025, 352, 2003)}]
    #
    # mean_width = 22
    # cluster2choice_m(cluster, mean_width, tf_box=True)


    # number_array = np.array([2, -1, 4, -1])
    # times = 0
    # number_list = infer_number(number_array, times, interval=1)
    # print(number_list)



    img_path = r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\11.jpg'
    image = cv2.imread(img_path)
    tf_sheet = [{'class_name': 'attention', 'bounding_box': {'xmin': 585, 'ymin': 407, 'xmax': 1516, 'ymax': 762}, 'score': '0.9986'}, {'class_name': 'choice', 'bounding_box': {'xmin': 133, 'ymin': 854, 'xmax': 1394, 'ymax': 1181}, 'score': '0.9986'}, {'class_name': 'choice', 'bounding_box': {'xmin': 131, 'ymin': 1267, 'xmax': 1330, 'ymax': 1371}, 'score': '0.9953'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 658, 'ymin': 837, 'xmax': 892, 'ymax': 1057}, 'score': '1.0000'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 187, 'ymin': 831, 'xmax': 423, 'ymax': 1055}, 'score': '1.0000'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 1133, 'ymin': 838, 'xmax': 1359, 'ymax': 1061}, 'score': '1.0000'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 194, 'ymin': 1275, 'xmax': 421, 'ymax': 1358}, 'score': '1.0000'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 661, 'ymin': 1099, 'xmax': 893, 'ymax': 1186}, 'score': '1.0000'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 189, 'ymin': 1098, 'xmax': 424, 'ymax': 1181}, 'score': '1.0000'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 659, 'ymin': 1272, 'xmax': 897, 'ymax': 1359}, 'score': '1.0000'}, {'class_name': 'choice_m', 'bounding_box': {'xmin': 1137, 'ymin': 1271, 'xmax': 1352, 'ymax': 1320}, 'score': '0.7672'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 147, 'ymin': 1274, 'xmax': 194, 'ymax': 1364}, 'score': '0.9996'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 615, 'ymin': 843, 'xmax': 663, 'ymax': 1056}, 'score': '0.9996'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 617, 'ymin': 1097, 'xmax': 664, 'ymax': 1192}, 'score': '0.9995'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 142, 'ymin': 837, 'xmax': 181, 'ymax': 1057}, 'score': '0.9989'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 1086, 'ymin': 838, 'xmax': 1136, 'ymax': 1062}, 'score': '0.9986'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 147, 'ymin': 1092, 'xmax': 194, 'ymax': 1186}, 'score': '0.9982'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 1086, 'ymin': 1269, 'xmax': 1132, 'ymax': 1327}, 'score': '0.9977'}, {'class_name': 'choice_n', 'bounding_box': {'xmin': 1091, 'ymin': 1098, 'xmax': 1136, 'ymax': 1154}, 'score': '0.9901'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 1133, 'ymin': 925, 'xmax': 1354, 'ymax': 969}, 'score': '0.9996'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 189, 'ymin': 923, 'xmax': 420, 'ymax': 967}, 'score': '0.9995'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 661, 'ymin': 924, 'xmax': 888, 'ymax': 967}, 'score': '0.9995'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 1138, 'ymin': 884, 'xmax': 1352, 'ymax': 926}, 'score': '0.9994'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 188, 'ymin': 838, 'xmax': 413, 'ymax': 878}, 'score': '0.9994'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 671, 'ymin': 1275, 'xmax': 891, 'ymax': 1315}, 'score': '0.9994'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 190, 'ymin': 967, 'xmax': 420, 'ymax': 1009}, 'score': '0.9993'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 664, 'ymin': 1010, 'xmax': 888, 'ymax': 1054}, 'score': '0.9989'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 668, 'ymin': 1141, 'xmax': 889, 'ymax': 1181}, 'score': '0.9989'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 666, 'ymin': 841, 'xmax': 890, 'ymax': 881}, 'score': '0.9988'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 1138, 'ymin': 1015, 'xmax': 1353, 'ymax': 1056}, 'score': '0.9988'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 190, 'ymin': 881, 'xmax': 416, 'ymax': 924}, 'score': '0.9986'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 1133, 'ymin': 971, 'xmax': 1354, 'ymax': 1014}, 'score': '0.9986'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 189, 'ymin': 1008, 'xmax': 417, 'ymax': 1053}, 'score': '0.9985'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 663, 'ymin': 882, 'xmax': 888, 'ymax': 925}, 'score': '0.9983'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 193, 'ymin': 1139, 'xmax': 416, 'ymax': 1181}, 'score': '0.9978'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 664, 'ymin': 1319, 'xmax': 892, 'ymax': 1361}, 'score': '0.9978'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 196, 'ymin': 1315, 'xmax': 414, 'ymax': 1356}, 'score': '0.9970'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 200, 'ymin': 1276, 'xmax': 411, 'ymax': 1312}, 'score': '0.9967'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 661, 'ymin': 968, 'xmax': 889, 'ymax': 1010}, 'score': '0.9963'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 664, 'ymin': 1102, 'xmax': 885, 'ymax': 1140}, 'score': '0.9932'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 1138, 'ymin': 844, 'xmax': 1351, 'ymax': 882}, 'score': '0.9788'}, {'class_name': 'choice_s', 'bounding_box': {'xmin': 193, 'ymin': 1102, 'xmax': 413, 'ymax': 1141}, 'score': '0.8138'}, {'class_name': 'class_w', 'bounding_box': {'xmin': 1033, 'ymin': 322, 'xmax': 1521, 'ymax': 389}, 'score': '0.9343'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 345, 'ymin': 400, 'xmax': 396, 'ymax': 746}, 'score': '0.9998'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 262, 'ymin': 403, 'xmax': 318, 'ymax': 748}, 'score': '0.9996'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 503, 'ymin': 400, 'xmax': 554, 'ymax': 742}, 'score': '0.9995'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 425, 'ymin': 398, 'xmax': 475, 'ymax': 744}, 'score': '0.9994'}, {'class_name': 'exam_number_s', 'bounding_box': {'xmin': 184, 'ymin': 398, 'xmax': 241, 'ymax': 745}, 'score': '0.9985'}, {'class_name': 'exam_number_w', 'bounding_box': {'xmin': 161, 'ymin': 353, 'xmax': 548, 'ymax': 404}, 'score': '0.9965'}, {'class_name': 'info_title', 'bounding_box': {'xmin': 423, 'ymin': 100, 'xmax': 1340, 'ymax': 259}, 'score': '0.9995'}, {'class_name': 'solve', 'bounding_box': {'xmin': 159, 'ymin': 1403, 'xmax': 1567, 'ymax': 1745}, 'score': '1.0000'}, {'class_name': 'solve', 'bounding_box': {'xmin': 160, 'ymin': 1788, 'xmax': 1568, 'ymax': 2203}, 'score': '0.9997'}, {'class_name': 'student_info', 'bounding_box': {'xmin': 592, 'ymin': 321, 'xmax': 1027, 'ymax': 389}, 'score': '0.9908'}, {'class_name': 'time', 'bounding_box': {'xmin': 310, 'ymin': 238, 'xmax': 432, 'ymax': 300}, 'score': '0.9526'}, {'class_name': 'total_score', 'bounding_box': {'xmin': 1083, 'ymin': 248, 'xmax': 1224, 'ymax': 305}, 'score': '0.9937'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 158, 'ymin': 1796, 'xmax': 313, 'ymax': 1824}, 'score': '0.9999'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 120, 'ymin': 802, 'xmax': 1214, 'ymax': 835}, 'score': '0.9998'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 155, 'ymin': 1413, 'xmax': 296, 'ymax': 1444}, 'score': '0.9996'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 101, 'ymin': 1192, 'xmax': 1564, 'ymax': 1227}, 'score': '0.9984'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 102, 'ymin': 1366, 'xmax': 649, 'ymax': 1401}, 'score': '0.9984'}, {'class_name': 'type_score', 'bounding_box': {'xmin': 707, 'ymin': 2257, 'xmax': 951, 'ymax': 2288}, 'score': '0.9952'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 145, 'ymin': 1787, 'xmax': 212, 'ymax': 1834}, 'score': '1.0000'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 70, 'ymin': 774, 'xmax': 155, 'ymax': 841}, 'score': '0.9997'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 153, 'ymin': 1399, 'xmax': 205, 'ymax': 1447}, 'score': '0.9997'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 77, 'ymin': 1187, 'xmax': 148, 'ymax': 1230}, 'score': '0.9996'}, {'class_name': 'type_score_n', 'bounding_box': {'xmin': 62, 'ymin': 1340, 'xmax': 148, 'ymax': 1413}, 'score': '0.9889'}, {'class_name': 'exam_number', 'bounding_box': {'xmin': 161, 'ymin': 398, 'xmax': 554, 'ymax': 748}}]
    infer_box_list = [{'loc': [83, 361, 1578, 454]}, {'loc': [83, 562, 1578, 748]}, {'loc': [83, 836, 1578, 1189]}, {'loc': [83, 1267, 1578, 1365]}, {'loc': [83, 1882, 1578, 1978]}]
    col_split_x = [1653]
    infer_choice_m_demo(image, tf_sheet, col_split_x, infer_box_list)