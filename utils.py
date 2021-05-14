# @Author  : lightXu
# @File    : utils.py
# @Time    : 2018/11/22 0022 上午 11:01
import os
import json
import math
import subprocess
import xml.etree.cElementTree as ET

import cv2, re
import numpy as np
from PIL import Image
from pywt import wavedec2, waverec2


def read_label(label_dir, subject):
    label_path = os.path.join(label_dir, "{}.txt".format(subject))
    label_list = ["__background__"]
    with open(label_path, "r", encoding="utf-8") as file:
        lines_list = file.readlines()
        for line in lines_list:
            line = line.strip().replace(" ", '').replace('，', ',').replace("'", "").replace('"', '').replace("\n", "")
            labels = [ele for ele in line.split(',') if ele]
            label_list += labels

    print("load {} labels: ".format(subject), len(label_list))
    return label_list


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def create_xml(obj_name, tree, xmin, ymin, xmax, ymax):
    root = tree.getroot()

    pobject = ET.SubElement(root, 'object', {})
    pname = ET.SubElement(pobject, 'name')
    pname.text = obj_name
    ppose = ET.SubElement(pobject, 'pose')
    ppose.text = 'Unspecified'
    ptruncated = ET.SubElement(pobject, 'truncated')
    ptruncated.text = '0'
    pdifficult = ET.SubElement(pobject, 'difficult')
    pdifficult.text = '0'
    # add bndbox
    pbndbox = ET.SubElement(pobject, 'bndbox')
    pxmin = ET.SubElement(pbndbox, 'xmin')
    pxmin.text = str(xmin)

    pymin = ET.SubElement(pbndbox, 'ymin')
    pymin.text = str(ymin)

    pxmax = ET.SubElement(pbndbox, 'xmax')
    pxmax.text = str(xmax)

    pymax = ET.SubElement(pbndbox, 'ymax')
    pymax.text = str(ymax)

    return tree


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bbox_list = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bbox_dict = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bbox_list.append(bbox_dict)

    serial = root.find('serial').text

    return serial, bbox_list


def read_xml_to_json(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    regions_list = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bbox_dict = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        region = {'class_name': class_name, 'bounding_box': bbox_dict}
        regions_list.append(region)

    sheet_dict = {'xml_path': xml_path, 'regions': regions_list}
    return sheet_dict


def crop_region(im, bbox):
    xmin = int(bbox['xmin'])
    ymin = int(bbox['ymin'])
    xmax = int(bbox['xmax'])
    ymax = int(bbox['ymax'])

    region = im[ymin:ymax, xmin:xmax]
    return region


def img_resize(analysis_type, im):
    min_size = 375
    max_size = 500
    if analysis_type == 'sheet':
        # min_size = 600
        # max_size = 800
        min_size = 1500
        max_size = 2000
    elif analysis_type == 'choice':
        min_size = 300
        max_size = 600
    elif analysis_type == 'choice_m':
        min_size = 600
        max_size = 600
    elif analysis_type == 'exam_number':
        min_size = 400
        max_size = 600
    elif analysis_type == 'cloze':
        min_size = 300
        max_size = 700
    elif analysis_type == 'solve':
        min_size = 300
        max_size = 700

    ycv, xcv = im.shape[0], im.shape[1]
    # cv2.imshow("image", im)
    # cv2.waitKey(100000)
    if ycv > xcv:
        # 使用cv2.resize时，参数输入是 宽×高×通道
        resize = cv2.resize(im, (min_size, max_size), interpolation=cv2.INTER_AREA)
        ratio = (float(xcv / min_size), float(ycv / max_size))
        return resize, ratio
    if ycv <= xcv:
        resize = cv2.resize(im, (max_size, min_size), interpolation=cv2.INTER_AREA)
        ratio = (float(xcv / max_size), float(ycv / min_size))
        return resize, ratio


def resize_faster_rcnn(analysis_type, im_orig):
    min_size = 1500
    max_size = 2000
    if analysis_type == 'math_blank':
        min_size = 1500
        max_size = 2000

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(min_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    return im, (1/im_scale, 1/im_scale)


def resize_by_percent(im, percent):
    """
    :param im:
    :param percent:
    :return: resize_img

    interpolation - 插值方法。共有5种：
    1)INTER_NEAREST - 最近邻插值法
    2)INTER_LINEAR - 双线性插值法（默认）
    3)INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。
      对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
    4)INTER_CUBIC - 基于4x4像素邻域的3次插值法
    5)INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值
    """

    height = im.shape[0]
    width = im.shape[1]
    new_x = int(width * percent)
    new_y = int(height * percent)

    res = cv2.resize(im, (new_x, new_y), interpolation=cv2.INTER_AREA)

    return res


def read_single_img(img_path):
    try:
        im = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    except FileNotFoundError as e:
        raise e
    return im


def write_single_img(dst, save_path):
    try:
        cv2.imencode('.jpg', dst)[1].tofile(save_path)
    except FileNotFoundError as e:
        raise e


def decide_coordinate_contains(coordinate1, coordinate2):
    xmin1 = coordinate1[0]
    ymin1 = coordinate1[1]
    xmax1 = coordinate1[2]
    ymax1 = coordinate1[3]
    mid_x = int(xmin1 + (xmax1 - xmin1)//2)
    mid_y = int(ymin1 + (ymax1 - ymin1)//2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]

    if xmin2 <= mid_x <= xmax2 and ymin2 <= mid_y <= ymax2:
        return True
    else:
        return False


def box_by_x_intervel(matrix, dif_length):
    """
    :param matrix: 选择题，考号等某一行所有unit的坐标
    :param dif_length: 字符间隔，小于此间隔，坐标合并
    :return: 合并后的所有unit的坐标
    """
    length, w = matrix.shape
    result_list = []
    if length == 1:
        result_list.append(matrix.tolist())  # 如果只有一个元素，那就直接添加
    else:
        i, j = 0, 0
        while i < length - 1:
            pre = matrix[i, :]
            rear = matrix[i+1, :]
            if (rear[0] - pre[2] <= dif_length) and i + 1 != length - 1:
                i += 1
            # 如果遍历的过程中发现下标i的值还没有到达倒数第二个下标
            # 那就继续前进
            elif (rear[0] - pre[2] <= dif_length) and i + 1 == length:
                lt = matrix[j:i + 1, :][:, :2].min(axis=0)
                rb = matrix[j:i + 1, :][:, 2:].max(axis=0)
                matrix_box = np.hstack([lt, rb]).tolist()
                result_list.append(matrix_box)
            # 如果发现下标i到达了倒数第二个下标了，为了避免程序执行
            # 循环条件终止而下标自动失效，不去记忆了，因为循环结束了，可是之前的下标
            # 没有完成切片操作。
            else:
                lt = matrix[j:i + 1, :][:, :2].min(axis=0)
                rb = matrix[j:i + 1, :][:, 2:].max(axis=0)
                matrix_box = np.hstack([lt, rb]).tolist()
                result_list.append(matrix_box)
                j = i + 1
                i += 1
            # 用j记下切片初始下标，用i来记录切片终止下标
            # 然后i和j自动加1，在满足条件的情况下继续前进
        if matrix[-1][0] - matrix[-2][2] <= dif_length:
            xmin = result_list[-1][0]
            xmax = matrix[-1][2]
            ymin = min(matrix[-2][1], matrix[-1][1])
            ymax = max(matrix[-2][3], matrix[-1][3])
            result_list[-1] = np.array([xmin, ymin, xmax, ymax]).tolist()
        else:
            result_list.append(matrix[-1].tolist())
        # 这个if-else语句是处理最后的一个元素，到底是独立成为一个切片
        # 还是并入到最后的切片中，根据条件判断即可

    return result_list


def check_qr_code_with_region_img(img_path):
    try:
        process = subprocess.check_output(
            ["zbarimg", '-q', img_path], shell=True,
            stderr=subprocess.STDOUT)
        result = process.decode('gbk')
        if result.find("QR-Code:") >= 0:
            info = result.replace('QR-Code:', '').replace('\r\n', '')
            return info
    except subprocess.CalledProcessError as e:
        # raise RuntimeError(
        #     "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output.decode('gbk')))
        info = 'Nan'
        return info


def crop_region_direct(im, bbox):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]

    region = im[ymin:ymax, xmin:xmax]
    return region


def decide_coordinate_contains1(coordinate1, coordinate2):
    xmin1 = coordinate1[0]
    ymin1 = coordinate1[1]
    xmax1 = coordinate1[2]
    ymax1 = coordinate1[3]
    mid_x = int(xmin1 + (xmax1 - xmin1)//2)
    mid_y = int(ymin1 + (ymax1 - ymin1)//2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]

    if xmin2 <= mid_x <= xmax2 and ymin2 <= mid_y <= ymax2:
        return True
    else:
        return False


def decide_coordinate_full_contains(coordinate1, coordinate2):
    xmin1 = coordinate1[0]
    ymin1 = coordinate1[1]
    xmax1 = coordinate1[2]
    ymax1 = coordinate1[3]
    mid_x = int(xmin1 + (xmax1 - xmin1)//2)
    mid_y = int(ymin1 + (ymax1 - ymin1)//2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]

    if xmin1 <= xmin2 and ymin1 <= ymin2 and xmax1 <= xmax2 and ymax1 <= ymax2:
        return True
    else:
        return False


def decide_coordinate_full_contains2(coordinate1, coordinate2):
    xmin1 = coordinate1[0]
    ymin1 = coordinate1[1]
    xmax1 = coordinate1[2]
    ymax1 = coordinate1[3]
    mid_x = int(xmin1 + (xmax1 - xmin1)//2)
    mid_y = int(ymin1 + (ymax1 - ymin1)//2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]

    if xmin1 <= xmin2 and ymin1 <= ymin2 and xmax1 >= xmax2 and ymax1 >= ymax2:
        return True
    else:
        return False


def decide_coordinate_left_top(coordinate1, coordinate2):
    coordinate11 = coordinate1[0]
    xmin1 = coordinate11[0]
    ymin1 = coordinate11[1]
    xmax1 = coordinate11[2]
    ymax1 = coordinate11[3]
    mid_x = int(xmin1 + (xmax1 - xmin1) // 2)
    mid_y = int(ymin1 + (ymax1 - ymin1) // 2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]

    if xmin1 <= xmin2:
        return 'l'
    elif ymin1 <= ymin2:
        return 't'
    else:
        return False


def decide_choice_m_left_top(digital, choice_m):
    xmin1 = digital[0]
    ymin1 = digital[1]
    xmax1 = digital[2]
    ymax1 = digital[3]
    mid_x = int(xmin1 + (xmax1 - xmin1) // 2)
    mid_y = int(ymin1 + (ymax1 - ymin1) // 2)

    xmin2 = choice_m[0]
    ymin2 = choice_m[1]
    xmax2 = choice_m[2]
    ymax2 = choice_m[3]

    if xmin1 <= xmin2 and ymin2 < mid_y < ymax2:
        return '180'
    elif ymin1 <= ymin2 and xmin2 < mid_x < xmax2:
        return '90'
    else:
        return '0'

# baidu
def decide_coordinate_left_baidu(coordinate1, coordinate2, x_y_interval_ave, singe_box_width_height_ave):
    xmin1 = coordinate1['left']
    ymin1 = coordinate1['top']
    xmax1 = coordinate1['left'] + coordinate1['width']
    ymax1 = coordinate1['top'] + coordinate1['height']
    mid_x1 = int(xmin1 + (xmax1 - xmin1) // 2)
    mid_y1 = int(ymin1 + (ymax1 - ymin1) // 2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]
    mid_x2 = int(xmin2 + (xmax2 - xmin2) // 2)
    mid_y2 = int(ymin2 + (ymax2 - ymin2) // 2)

    if xmin1 <= xmax2 and abs(mid_x1 - xmin2) <= x_y_interval_ave[0] + singe_box_width_height_ave[0] \
            and ymin2 <= mid_y1 <= ymax2 and xmin1 > xmin2 - x_y_interval_ave[0] - 3 * singe_box_width_height_ave[0]:
        return True
    else:
        return False


# baidu
def decide_coordinate_top_baidu(coordinate1, coordinate2, x_y_interval_ave, singe_box_width_height_ave):
    xmin1 = coordinate1['left']
    ymin1 = coordinate1['top']
    xmax1 = coordinate1['left'] + coordinate1['width']
    ymax1 = coordinate1['top'] + coordinate1['height']
    mid_x1 = int(xmin1 + (xmax1 - xmin1) // 2)
    mid_y1 = int(ymin1 + (ymax1 - ymin1) // 2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]
    mid_x2 = int(xmin2 + (xmax2 - xmin2) // 2)
    mid_y2 = int(ymin2 + (ymax2 - ymin2) // 2)

    if ymin1 <= ymax2 and abs(mid_y1 - ymin2) <= x_y_interval_ave[1] + singe_box_width_height_ave[1] \
            and xmin2 <= mid_x1 <= xmax2 and ymin1 > ymin2 - x_y_interval_ave[1] - 3 * singe_box_width_height_ave[1]:
        return True
    else:
        return False


# tr
def decide_coordinate_left(coordinate1, coordinate2, x_y_interval_ave, singe_box_width_height_ave):
    xmin1 = coordinate1['left']
    ymin1 = coordinate1['top']
    xmax1 = coordinate1['left'] + coordinate1['width']
    ymax1 = coordinate1['top'] + coordinate1['height']


    mid_x1 = int(xmin1 + (xmax1 - xmin1) // 2)
    mid_y1 = int(ymin1 + (ymax1 - ymin1) // 2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]
    mid_x2 = int(xmin2 + (xmax2 - xmin2) // 2)
    mid_y2 = int(ymin2 + (ymax2 - ymin2) // 2)

    if xmin1 <= xmin2 and int(4/5 * abs(mid_x1 - xmin2)) <= x_y_interval_ave[0] + singe_box_width_height_ave[0] \
            and ymin2 <= mid_y1 <= ymax2 and \
            xmin1 > xmin2 - x_y_interval_ave[0] - int(3 * singe_box_width_height_ave[0]):
        return True
    else:
        return False

# tr
def decide_coordinate_top(coordinate1, coordinate2, x_y_interval_ave, singe_box_width_height_ave):
    xmin1 = coordinate1['left']
    ymin1 = coordinate1['top']
    xmax1 = coordinate1['left'] + coordinate1['width']
    ymax1 = coordinate1['top'] + coordinate1['height']
    mid_x1 = int(xmin1 + (xmax1 - xmin1) // 2)
    mid_y1 = int(ymin1 + (ymax1 - ymin1) // 2)

    xmin2 = coordinate2[0]
    ymin2 = coordinate2[1]
    xmax2 = coordinate2[2]
    ymax2 = coordinate2[3]
    mid_x2 = int(xmin2 + (xmax2 - xmin2) // 2)
    mid_y2 = int(ymin2 + (ymax2 - ymin2) // 2)

    if ymin1 <= ymax2 and int(4/5 * abs(mid_y1 - ymin2)) <= x_y_interval_ave[1] + singe_box_width_height_ave[1] \
            and xmin2 <= mid_x1 <= xmax2 and ymin1 > \
            ymin2 - x_y_interval_ave[1] - int(3 * singe_box_width_height_ave[1]):
        return True
    else:
        return False


def combine_char(all_digital_list):
    new_all_digital_list = []
    i = 1
    while i <= len(all_digital_list):
        pre_one = all_digital_list[i - 1]
        if i == len(all_digital_list):
            new_all_digital_list.append(pre_one)
            break
        rear_one = all_digital_list[i]
        condition1 = abs(pre_one['location']['top'] - rear_one['location']['top']) < pre_one['location'][
            'height']  # 两字高度差小于一字高度
        condition2 = pre_one['location']['left'] + 1.5 * pre_one['location']['width'] > rear_one['location'][
            'left']  # 某字宽度的2倍大于两字间间隔
        if condition1:
            if condition2:
                new_char = pre_one['char'] + rear_one['char']
                new_location = {'left': pre_one['location']['left'],
                                'top': min(pre_one['location']['top'], rear_one['location']['top']),
                                'width': rear_one['location']['left'] + rear_one['location']['width'] -
                                         pre_one['location']['left'],
                                'height': max(pre_one['location']['height'], rear_one['location']['height'])}
                new_all_digital_list.append({'char': new_char, 'location': new_location})
                i = i + 1 + 1
            else:
                new_all_digital_list.append(pre_one)
                i = i + 1
        else:
            new_all_digital_list.append(pre_one)  # 遇到字符y轴相差过大就结束
            i = i + 1
    return new_all_digital_list


def change_baidu_to_tr_format(char_list):
    tr_format_list = []
    for ele in char_list:
        location = {}
        box = ele[0][: 4]
        char = ele[1]

        location['left'] = round(box[0]) - round(1 / 2 * box[2])
        location['top'] = round(box[1]) - round(1 / 2 * box[3])
        location['width'] = round(box[2])
        location['height'] = round(box[3])

        tr_format_list.append({'char': char, 'location': location})
    return tr_format_list


def get_img_region_box0(s_box, b_box):
    s_xmin = s_box[0]
    s_ymin = s_box[1]
    s_xmax = s_box[2]
    s_ymax = s_box[3]

    b_xmin = b_box[0]
    b_ymin = b_box[1]
    b_xmax = b_box[2]
    b_ymax = b_box[3]

    xmin = int(abs(b_xmin + s_xmin))
    ymin = int(abs(b_ymin + s_ymin))
    xmax = int(abs(s_xmax + b_xmin))
    ymax = int(abs(s_ymax + b_ymin))

    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def get_img_region_box01(s_box, b_box):
    s_xmin = s_box[0]
    s_ymin = s_box[1]
    s_xmax = s_box[2]
    s_ymax = s_box[3]

    b_xmin = b_box['xmin']
    b_ymin = b_box['ymin']
    b_xmax = b_box['xmax']
    b_ymax = b_box['ymax']

    xmin = int(abs(b_xmin + s_xmin))
    ymin = int(abs(b_ymin + s_ymin))
    xmax = int(abs(s_xmax + b_xmin))
    ymax = int(abs(s_ymax + b_ymin))

    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def get_img_region_box1(choice_s_box, choice_box):
    # height, width = im.shape
    choice_s_xmin = choice_s_box[0]
    choice_s_ymin = choice_s_box[1]
    choice_s_xmax = choice_s_box[2]
    choice_s_ymax = choice_s_box[3]

    choice_xmin = choice_box[0]
    choice_ymin = choice_box[1]
    choice_xmax = choice_box[2]
    choice_ymax = choice_box[3]

    xmin = int(abs(choice_xmin - choice_s_xmin))
    ymin = int(abs(choice_ymin - choice_s_ymin))
    xmax = int(abs(choice_s_xmax - choice_xmin))
    ymax = int(abs(choice_s_ymax - choice_ymin))

    region = [xmin, ymin, xmax, ymax]
    # region = im[ymin:ymax, xmin:xmax]
    return region


def get_min_distance(coordinate1, coordinate2):  # 欧式距离最小值

    def dist(point1, point2):
        distance = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        return math.sqrt(distance)

    (x1, y1, x1b, y1b) = coordinate1
    (x2, y2, x2b, y2b) = coordinate2
    left = x2b < x1  # 2在1的坐标左边
    right = x1b < x2  # 2在1的坐标右边
    bottom = y2b < y1  # 2在1的坐标下边
    top = y1b < y2  # 2在1的坐标上边
    if top and left:
        return dist((x1, y1b), (x2b, y2)), 'c'
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b)), 'c'
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b)), 'c'
    elif right and top:
        return dist((x1b, y1b), (x2, y2)), 'c'
    elif left:
        return 'w'
    elif right:
        return 'w'
    elif bottom:
        return y1 - y2b, 'h'
    elif top:
        return y2 - y1b, 'h'
    else:  # rectangles intersect
        return 'i'


def get_x_diff_and_y_diff1(single_choice_m_coordinates, cols):
    single_choice_m_coordinates_x = sorted(single_choice_m_coordinates, key=lambda k: k[0])
    single_choice_m_matrix_x = np.array(single_choice_m_coordinates_x)
    x_diff = single_choice_m_matrix_x[1:, 0] - single_choice_m_matrix_x[:-1, 2]
    x_diff_ = [ele for ele in x_diff.tolist() if ele < 0]
    xx = [ele for ele in x_diff.tolist() if ele not in x_diff_]
    x_dif_length = int(np.mean(xx))

    single_choice_m_coordinates_y = sorted(single_choice_m_coordinates, key=lambda k: k[1])
    single_choice_m_matrix_y = np.array(single_choice_m_coordinates_y)
    if len(single_choice_m_coordinates_y) == cols:
        y_dif_length = 2 * int(np.mean(single_choice_m_matrix_y[:, 3] - single_choice_m_matrix_y[:, 1])) // 3
    else:
        yy_diff = single_choice_m_matrix_y[1:, 1] - single_choice_m_matrix_y[:-1, 3]
        y_diff_ = [ele for ele in yy_diff.tolist() if ele < 0]
        yy = [ele for ele in yy_diff.tolist() if ele not in y_diff_]
        if len(yy) == 0:
            y_dif_length = 'nan'
        else:
            y_dif_length = int(np.mean(yy))
    x_y_interval = (x_dif_length, y_dif_length)
    return x_y_interval


def get_x_diff_and_y_diff(single_choice_m_coordinates):
    single_choice_m_coordinates_x = sorted(single_choice_m_coordinates, key=lambda k: k[0])
    single_choice_m_matrix_x = np.array(single_choice_m_coordinates_x)
    x_diff = single_choice_m_matrix_x[1:, 0] - single_choice_m_matrix_x[:-1, 2]
    x_diff_ = [ele for ele in x_diff.tolist() if ele < 0]
    xx = [ele for ele in x_diff.tolist() if ele not in x_diff_]
    x_dif_length = int(np.mean(xx))

    single_choice_m_coordinates_y = sorted(single_choice_m_coordinates, key=lambda k: k[1])
    single_choice_m_matrix_y = np.array(single_choice_m_coordinates_y)
    yy_diff = single_choice_m_matrix_y[1:, 1] - single_choice_m_matrix_y[:-1, 3]
    y_diff_ = [ele for ele in yy_diff.tolist() if ele < 0]
    yy = [ele for ele in yy_diff.tolist() if ele not in y_diff_]
    y_dif_length = int(np.mean(yy))
    x_y_interval = (x_dif_length, y_dif_length)
    return x_y_interval


def list_to_dict(box_list):
    location_s_box = {}
    location_s_box['xmin'] = box_list[0]
    location_s_box['ymin'] = box_list[1]
    location_s_box['xmax'] = box_list[2]
    location_s_box['ymax'] = box_list[3]
    return location_s_box


def infer_number(number_list, times=0, interval=1):
    if times > 30:
        return number_list
    # 默认题号间隔为1
    if number_list[-1] != -1 and len(list(set(number_list[:-1]))) == 1:
        new_number_list = []
        for k in range(1, len(number_list)):
            number = number_list[-1] - k
            new_number_list.append(number)
        new_number_list.append(number_list[-1])
        return sorted(new_number_list)
    elif -1 not in number_list or sum(number_list) == -1 * len(number_list):
        return number_list
    else:
        for n_index in range(0, len(number_list) - 1):
            if n_index == 0:
                if number_list[n_index] != -1:

                    if len(number_list) > 1 and number_list[n_index + 1] == -1:
                        number_list[n_index + 1] = number_list[n_index] + interval

            if n_index != 0 and number_list[n_index] != -1:
                if number_list[n_index - 1] == -1:
                    number_list[n_index - 1] = number_list[n_index] - interval
                if number_list[n_index + 1] == -1:
                    number_list[n_index + 1] = number_list[n_index] + interval
        times += 1
        return infer_number(number_list, times)


def combine_char_in_raw_format(word_result_list, left_boundary=0, top_boundary=0):
    new_all_word_list = []
    for index, chars_dict in enumerate(word_result_list):
        chars_list = chars_dict['chars']
        chars_box_list = []
        char_str = ''
        for ele in chars_list:
            location = ele['location']
            left, top, width, height = location['left']+left_boundary, location['top']+top_boundary, location['width'], location['height']
            right, bottom = left + width, top + height

            box = (left, top, right, bottom, width, height)
            chars_box_list.append(box)
            char_str = char_str + ele['char']

        split_index = [0]
        for i in range(1, len(chars_box_list)):
            pre = chars_box_list[i - 1]
            crt = chars_box_list[i]
            x_dif = crt[0] - pre[2]
            y_dif = crt[1] - pre[1]

            if y_dif < pre[5]:
                if x_dif < 1 * pre[4]:
                    pass
                else:
                    split_index.append(i)
            else:
                split_index.append(i)

        split_index.append(len(chars_box_list))

        combine_str_list = []
        for i in range(1, len(split_index)):
            combine_str = char_str[split_index[i - 1]:split_index[i]]
            location_arr = np.asarray(chars_box_list[split_index[i - 1]:split_index[i]])
            min_arr = location_arr.min(axis=0)
            max_arr = location_arr.max(axis=0)
            location = {'left': min_arr[0], 'top': min_arr[1],
                        'width': max_arr[2] - min_arr[0], 'height': max_arr[3] - min_arr[1]}
            new_chars_list = []
            for ii, loc in enumerate(location_arr):
                char = combine_str[ii]
                ll = loc[0]
                tt = loc[1]
                ww = loc[4]
                hh = loc[5]
                new_chars_list.append({'char': char,
                                       'location': {'left': ll, 'top': tt, 'width': ww, 'height': hh}})

            combine_str_list.append({'words': combine_str,
                                     'location': location,
                                     'chars': new_chars_list})

        new_all_word_list = new_all_word_list + combine_str_list

    return new_all_word_list


def cal_iou(box1, box2, box_type='dict'):
    if box_type == 'list':
        in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
        in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
        inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = (box1_area + box2_area - inter)

        res = inter / union
        res1 = inter / box1_area
        res2 = inter / box2_area
    else:
        in_h = min(box1['xmax'], box2['xmax']) - max(box1['xmin'], box2['xmin'])
        in_w = min(box1['ymax'], box2['ymax']) - max(box1['ymin'], box2['ymin'])
        inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
        box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
        box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
        union = ((box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin']) +
                 (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin']) - inter)
        res = inter / union
        res1 = inter / box1_area
        res2 = inter / box2_area

    return res, res1, res2


def _binary_array_to_hex(arr):
    """
    internal function to make a hex string out of a binary array.
    """
    bit_string = ''.join(str(b) for b in 1 * arr.flatten())
    width = int(np.ceil(len(bit_string) / 4))
    return '{:0>{width}x}'.format(int(bit_string, 2), width=width)


class ImageHash(object):
    """
    Hash encapsulation. Can be used for dictionary keys and comparisons.
    """

    def __init__(self, binary_array):
        self.hash = binary_array

    def __str__(self):
        return _binary_array_to_hex(self.hash.flatten())

    def __repr__(self):
        return repr(self.hash)

    def __sub__(self, other):
        if other is None:
            raise TypeError('Other hash must not be None.')
        if self.hash.size != other.hash.size:
            raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
        return np.count_nonzero(self.hash.flatten() != other.hash.flatten())

    def __eq__(self, other):
        if other is None:
            return False
        return np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __ne__(self, other):
        if other is None:
            return False
        return not np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __hash__(self):
        # this returns a 8 bit integer, intentionally shortening the information
        return sum([2 ** (i % 8) for i, v in enumerate(self.hash.flatten()) if v])


def whash(image, hash_size=8, image_scale=None, mode='haar', remove_max_haar_ll=True):
    if image_scale is not None:
        assert image_scale & (image_scale - 1) == 0, "image_scale is not power of 2"
    else:
        image_natural_scale = 2 ** int(np.log2(min(image.size)))
        image_scale = max(image_natural_scale, hash_size)

    ll_max_level = int(np.log2(image_scale))

    level = int(np.log2(hash_size))
    assert hash_size & (hash_size - 1) == 0, "hash_size is not power of 2"
    assert level <= ll_max_level, "hash_size in a wrong range"
    dwt_level = ll_max_level - level

    image = image.resize((image_scale, image_scale), Image.ANTIALIAS)
    pixels = np.asarray(image) / 255

    # Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
    if remove_max_haar_ll:
        coeffs = wavedec2(pixels, 'haar', level=ll_max_level)
        coeffs = list(coeffs)
        coeffs[0] *= 0
        pixels = waverec2(coeffs, 'haar')

    # Use LL(K) as freq, where K is log2(@hash_size)
    coeffs = wavedec2(pixels, mode, level=dwt_level)
    dwt_low = coeffs[0]

    # Substract median and compute hash
    med = np.median(dwt_low)
    diff = dwt_low > med
    return ImageHash(diff)


def average_hash(image, hash_size):
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    # reduce size and complexity, then covert to grayscale
    image = image.resize((hash_size, hash_size), Image.ANTIALIAS)
    # pixels = resize_by_fixed_size(image, hash_size[0], hash_size[1])

    # find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
    pixels = np.asarray(image)
    avg = pixels.mean()

    # create string of bits
    diff = pixels > avg
    # make a hash
    return ImageHash(diff)


def hash_similarity(hash_as_t, hash_to_regi):
    similarity = 1 - (hash_as_t - hash_to_regi) / len(hash_as_t.hash) ** 2  # 相似性
    return similarity


def image_hash_detection_simple(image1, image2):
    # w1 = phash(Image.fromarray(image1), image_scale=64, hash_size=8, mode='db4')
    # w2 = phash(Image.fromarray(image2), image_scale=64, hash_size=8, mode='db4')

    w1 = average_hash(Image.fromarray(image1), hash_size=8)
    w2 = average_hash(Image.fromarray(image2), hash_size=8,)

    simi = hash_similarity(w1, w2)
    return simi


def xyxy2xywh(bbox_list):
    bbox_arr = np.array(bbox_list)
    bbox_array = bbox_arr.copy()
    bbox_array[:, 0] = 1/2 * (bbox_arr[:, 0] + bbox_arr[:, 2])
    bbox_array[:, 1] = 1/2 * (bbox_arr[:, 1] + bbox_arr[:, 3])
    bbox_array[:, 2] = 1/2 * (bbox_arr[:, 2] - bbox_arr[:, 0])
    bbox_array[:, 3] = 1/2 * (bbox_arr[:, 3] - bbox_arr[:, 1])
    return bbox_array


def xywh2xyxy(bbox_list):
    bbox_arr = np.array(bbox_list)
    bbox_array = bbox_arr.copy()
    bbox_array[:, 0] = bbox_arr[:, 0] - 1/2 * bbox_arr[:, 2]
    bbox_array[:, 1] = bbox_arr[:, 1] - 1/2 * bbox_arr[:, 3]
    bbox_array[:, 2] = bbox_arr[:, 0] + 1/2 * bbox_arr[:, 2]
    bbox_array[:, 3] = bbox_arr[:, 1] + 1/2 * bbox_arr[:, 3]
    return bbox_array


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def preprocess(img, binary_inv=False):
    dilate = 1
    blur = 1

    if len(img.shape) >= 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # # Apply dilation and erosion to remove some noise
    if dilate != 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        img = cv2.dilate(gray_img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

    # Apply blur to smooth out the edges
    if blur != 0:
        img = cv2.GaussianBlur(img, (blur, blur), 0)

    # Apply threshold to get image with only b&w (binarization)
    if binary_inv:
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    else:
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return img
