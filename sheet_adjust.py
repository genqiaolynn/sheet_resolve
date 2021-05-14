# @Author  : mbq
# @File    : sheet_adjust.py
# @Time    : 2019/9/26 0026 上午 10:12
import copy
import json
import os

import cv2
import numpy as np

''' 根据CV检测矩形框 调整模型输出框'''
''' LSD直线检测 暂时改用 霍夫曼检测'''

ADJUST_CLASS = ['solve', 'solve0', 'composition', 'composition0', 'choice', 'cloze', 'correction']


# 用户自己计算阈值
def custom_threshold(gray, type_inv=cv2.THRESH_BINARY):
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
    # h, w = gray.shape[:2]
    # m = np.reshape(gray, [1, w * h])
    # mean = m.sum() / (w * h)
    mean = np.mean(gray)
    ret, binary = cv2.threshold(gray, min(230, mean), 255, type_inv)
    return binary


# 开运算
def open_img(image_bin, kera=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kera)
    opening = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel)
    return opening


# 闭运算
def close_img(image_bin, kera=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kera)
    closing = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel)
    return closing


# 腐蚀
def erode_img(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(image, kernel)
    return erosion


# 膨胀
def dilation_img(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilaion = cv2.dilate(image, kernel)
    return dilaion


# 图像padding
def image_padding(image, padding_w, padding_h):
    h, w = image.shape[:2]
    if 3 == len(image.shape):
        image_new = np.zeros((h + padding_h, w + padding_w, 3), np.uint8)
    else:
        image_new = np.zeros((h + padding_h, w + padding_w), np.uint8)
    image_new[int(padding_h / 2):int(padding_h / 2) + h, int(padding_w / 2):int(padding_w / 2) + w] = image
    return image_new


def horizontal_projection(img_bin, mut=0):
    """水平方向投影"""
    h, w = img_bin.shape[:2]
    hist = [0 for i in range(w)]
    for x in range(w):
        tmp = 0
        for y in range(h):
            if img_bin[y][x]:
                tmp += 1
        if tmp > mut:
            hist[x] = tmp
    return hist


def vertical_projection(img_bin, mut=0):
    """垂直方向投影"""
    h, w = img_bin.shape[:2]
    hist = [0 for i in range(h)]
    for y in range(h):
        tmp = 0
        for x in range(w):
            if img_bin[y][x]:
                tmp += 1
        if tmp > mut:
            hist[y] = tmp
    return hist


def get_white_blok_pos(arry, blok_w=0):
    """获取投影结果中的白色块"""
    pos = []
    start = 1
    x0 = 0
    x1 = 0
    for idx, val in enumerate(arry):
        if start:
            if val:
                x0 = idx
                start = 0
        else:
            if 0 == val:
                x1 = idx
                start = 1
                if x1 - x0 > blok_w:
                    pos.append((x0, x1))
    if 0 == start:
        x1 = len(arry) - 1
        if x1 - x0 > blok_w:
            pos.append((x0, x1))
    return pos


def get_decide_boberLpa(itemRe, itemGT):
    """
    IOU 计算
    """
    x1 = int(itemRe[0])
    y1 = int(itemRe[1])
    x1_ = int(itemRe[2])
    y1_ = int(itemRe[3])
    width1 = x1_ - x1
    height1 = y1_ - y1

    x2 = int(float(itemGT[0]))
    y2 = int(float(itemGT[1]))
    x2_ = int(float(itemGT[2]))
    y2_ = int(float(itemGT[3]))
    width2 = x2_ - x2
    height2 = y2_ - y2

    endx = max(x1_, x2_)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1_, y2_)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    AreaJc = 0
    ratio = 0.0

    if width <= 0 or height <= 0:
        res = 0
    else:
        AreaJc = width * height
        AreaRe = width1 * height1
        AreaGT = width2 * height2
        ratio = float(AreaJc) / float((AreaGT + AreaRe - AreaJc))
    return ratio


# 查找连通区域 微调专用 不通用

def get_contours(image):
    # image = cv2.imread(img_path,0)
    # if debug: plt_imshow(image)
    image_binary = custom_threshold(image)
    # if debug: plt_imshow(image_binary)
    # if debug: cv2.imwrite(os.path.join(file_dir,"bin.jpg"),image_binary)
    image_dilation = open_img(image_binary, kera=(5, 1))
    image_dilation = open_img(image_dilation, kera=(1, 5))
    # if debug: plt_imshow(image_dilation)
    # if debug: cv2.imwrite(os.path.join(file_dir,"dia.jpg"),image_dilation)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(image_dilation)
    rects = []
    img_h, img_w = image.shape[:2]
    for box in stats:
        x0 = int(box[0])
        y0 = int(box[1])
        w = int(box[2])
        h = int(box[3])
        area = int(box[4])
        if w < img_w / 5 or w > img_w - 10 or h < 50 or h > img_h - 10:  # 常见框大小限定
            continue
        if img_w > img_h:  # 多栏答题卡 w大于宽度的一般肯定是错误的框
            if w > img_w / 2:
                continue
        if area < w * h / 3:  # 大框套小框 中空白色区域形成的面积 排除
            continue
        rects.append((x0, y0, x0 + w, y0 + h))
    return rects


def adjust_alarm_info(image, box):
    """
    调整上下坐标 排除内部含有了边框线情况
    左右调整只有100%确认的 从边界开始遇到的第一个非0列就终止 误伤情况太多
    LSD算法转不过来  霍夫曼检测不靠谱 连通区域测试后排除误伤情况太多  改用投影
    image: 灰度 非 二值图
    box  : 坐标信息
    """
    # debug
    # debug = 0

    if image is None:
        print("error image")
        return box
    img_box = image[box[1]:box[3], box[0]:box[2]]
    h, w = img_box.shape[:2]

    # debug
    # if debug: ia.imshow(img_box)

    img_bin = custom_threshold(img_box, type_inv=cv2.THRESH_BINARY_INV)
    img_padding = image_padding(img_bin, 100, 100)
    img_close = close_img(img_padding, kera=(30, 3))
    img_back = img_close[50:50 + h, 50:50 + w]

    # debug
    # if debug: ia.imshow(img_back)

    # 垂直投影 找 left top
    hist_vert = vertical_projection(img_back, mut=h / 4)

    # debug
    # if debug:
    #     print(hist_vert)
    #     black_img_h = np.zeros_like(img_back)
    #     for idx, val in enumerate(hist_vert):
    #         if (val == 0):
    #             continue
    #         for x in range(val):
    #             black_img_h[idx][x] = 255
    #     ia.imshow(black_img_h)

    y_pos = get_white_blok_pos(hist_vert, 2)
    if (len(y_pos) == 0):
        return box

    # 获取最大的作为alarm_info的区域
    max_id = 0
    max_len = 0
    for idx, pos_tmp in enumerate(y_pos):
        pos_len = abs(pos_tmp[1] - pos_tmp[0])
        if (pos_len > max_len):
            max_id = idx
            max_len = pos_len

    # debug to show
    # if debug:
    #     img_show = cv2.cvtColor(img_box, cv2.COLOR_GRAY2BGR)
    #     cv2.line(img_show, (0, y_pos[max_id][0]), (w - 1, y_pos[max_id][0]), (0, 0, 255), 2)
    #     cv2.line(img_show, (0, y_pos[max_id][1]), (w - 1, y_pos[max_id][1]), (0, 0, 255), 2)
    #     ia.imshow(img_show)

    # 左右 的微调
    img_next = img_bin[y_pos[max_id][0]:y_pos[max_id][1], 0:w - 1]
    img_lr_close = open_img(img_next, kera=(1, 1))
    img_lr_close = close_img(img_lr_close, kera=(3, 1))

    # debug
    # if debug: ia.imshow(img_lr_close)

    hist_proj = horizontal_projection(img_lr_close, mut=1)
    w_len = len(hist_proj)
    new_left = 0
    new_right = w_len - 1
    b_flag = [0, 0]
    for idx, val in enumerate(hist_proj):
        if 0 == b_flag[0]:
            if val != 0:
                new_left = idx
                b_flag[0] = 1
        if 0 == b_flag[1]:
            if hist_proj[w_len - 1 - idx] != 0:
                new_right = w_len - idx - 1
                b_flag[1] = 1
        if b_flag[0] and b_flag[1]:
            break

    new_top = box[1] + y_pos[max_id][0]
    new_bottom = box[1] + y_pos[max_id][1]
    new_left += box[0]
    new_right += box[0]
    box[1] = new_top
    box[3] = new_bottom
    box[0] = new_left
    box[2] = new_right

    return box


def adjust_zg_info(image, box, cv_boxes, name):
    """
    调整大区域的box
    1、cvbox要与box纵坐标有交叉
    2、IOU值大于0。8时 默认相等拷贝区域坐标
    """
    if image is None:
        return box

    min_rotio = 0.5
    img_box = image[box[1]:box[3], box[0]:box[2]]
    cv2.imwrite('data/'+name + '.jpg', img_box)
    h, w = img_box.shape[:2]

    jc_boxes = []  # 记录与box存在交叉的 cv_boxes
    tmp_rotio = 0
    rc_mz = box
    for idx, cv_box in enumerate(cv_boxes):
        if (box[1] - 10) > (cv_box[3]):  # 首先要保证纵坐标有交叉
            continue
        if (box[3] + 10) < cv_box[1]:
            continue

        jc_x = max(box[0], cv_box[0])
        jc_y = min(box[2], cv_box[2])
        bj_x = min(box[0], cv_box[0])
        bj_y = max(box[2], cv_box[2])

        rt = abs(jc_y - jc_x) * 1.0 / abs(bj_y - bj_x) * 1.0
        if rt < min_rotio:
            continue
        jc_boxes.append(cv_box)
        if rt > tmp_rotio:
            rc_mz = cv_box
            tmp_rotio = rt
    # 判断 调整
    if len(jc_boxes) != 0:
        box[0] = rc_mz[0]
        box[2] = rc_mz[2]
        b_find = 0
        frotio = 0.0
        rc_biggst = rc_mz
        for mz_box in jc_boxes:
            iou = get_decide_boberLpa(mz_box, box)
            if iou > 0.8:
                b_find = 1
                frotio = iou
                rc_biggst = mz_box
        if b_find:
            box[1] = rc_biggst[1]
            box[3] = rc_biggst[3]
    return box


def adjust_item_edge(img_path, reback_json):
    """
    根据图像的CV分析结果和 模型直接输出结果 对模型输出的边框做微调
    1、外接矩形查找
    2、LSD直线检测 替换方法 霍夫曼直线检测
    3、只处理有把握的情况 任何含有不确定因素的一律不作任何处理
    img_path: 待处理图像绝对路径
    re_json : 模型输出结果
    """
    debug = 1
    # 存放新的结果
    re_json = copy.deepcopy(reback_json)
    if not os.path.exists(img_path) or 0 == len(re_json):
        return
    image = cv2.imread(img_path, 0)
    # 获取CV连通区域结果
    cv_boxes = get_contours(image)

    if debug:
        print(len(cv_boxes))
        image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # for item in cv_boxes:
        #     cv2.rectangle(image_draw, (item[0], item[1]), (item[2], item[3]), (0, 0, 250), 2)
        # cv2.imwrite(os.path.join(file_dir, "show.jpg"), image_draw)
    # 循环处理指定的box
    for idx, item in enumerate(re_json):
        name = item["class_name"]
        box = [item["bounding_box"]["xmin"], item["bounding_box"]["ymin"], item["bounding_box"]["xmax"],
               item["bounding_box"]["ymax"]]
        # print(name ,box)
        if name == "alarm_info" or name == "page" or name == "type_score":
            if debug:
                cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            new_box = adjust_alarm_info(image, box)
            if debug:
                cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            item["bounding_box"]["xmin"] = box[0]
            item["bounding_box"]["xmax"] = box[2]
            item["bounding_box"]["ymin"] = box[1]
            item["bounding_box"]["ymax"] = box[3]
        elif (name == "solve" or name == "solve0"
              or name == "cloze" or name == "choice"
              or name == "composition" or name == "composition0"):
            if debug:
                cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            new_box = adjust_zg_info(image, box, cv_boxes)
            if debug:
                cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            item["bounding_box"]["xmin"] = box[0]
            item["bounding_box"]["xmax"] = box[2]
            item["bounding_box"]["ymin"] = box[1]
            item["bounding_box"]["ymax"] = box[3]
        else:
            pass
    if debug:
        cv2.imwrite(os.path.join(r"E:\data\aug_img\adjust", "show.jpg"), image_draw)
    return re_json


def adjust_item_edge_by_gray_image(image, reback_json):
    '''
    根据图像的CV分析结果和 模型直接输出结果 对模型输出的边框做微调
    1、外接矩形查找
    2、LSD直线检测 替换方法 霍夫曼直线检测
    3、只处理有把握的情况 任何含有不确定因素的一律不作任何处理
    img_path: 待处理图像绝对路径
    re_json : 模型输出结果
    '''

    # regions1 = reback_json['regions']
    # for ele in regions1:
    #     cv2.rectangle(image, (ele["bounding_box"]["xmin"], ele["bounding_box"]["ymin"]),
    #                   (ele["bounding_box"]["xmax"], ele["bounding_box"]["ymax"]), (0, 255, 0), 2)
    # cv2.imwrite(os.path.join('data', "raw.jpg"), image)

    debug = 0
    re_json = copy.deepcopy(reback_json)
    # 存放新的结果
    # 获取CV连通区域结果
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv_boxes = get_contours(image)

    # draw box
    # for ele in cv_boxes:
    #     cv2.rectangle(image, (ele[0], ele[1]), (ele[2], ele[3]), (0, 255, 0), 2)
    # cv2.imwrite('data/save.jpg', image)
    # print('draw_box')

    if debug:
        # print(len(cv_boxes))
        image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # for item in cv_boxes:
        #     cv2.rectangle(image_draw, (item[0], item[1]), (item[2], item[3]), (0, 0, 250), 2)
        # cv2.imwrite(os.path.join(file_dir, "show.jpg"), image_draw)
    # 循环处理指定的box
    regions = re_json['regions']
    for idx, item in enumerate(regions):
        name = item["class_name"]
        box = [item["bounding_box"]["xmin"], item["bounding_box"]["ymin"], item["bounding_box"]["xmax"],
               item["bounding_box"]["ymax"]]
        # print(name ,box)
        if name == "alarm_info" or name == "page" or name == "type_score":
            if debug:
                cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            new_box = adjust_alarm_info(image, box)
            if debug:
                cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            item["bounding_box"]["xmin"] = box[0]
            item["bounding_box"]["xmax"] = box[2]
            item["bounding_box"]["ymin"] = box[1]
            item["bounding_box"]["ymax"] = box[3]
        elif name in ADJUST_CLASS:
            if debug:
                cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            new_box = adjust_zg_info(image, box, cv_boxes, name)
            if debug:
                cv2.rectangle(image_draw, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            item["bounding_box"]["xmin"] = box[0]
            item["bounding_box"]["xmax"] = box[2]
            item["bounding_box"]["ymin"] = box[1]
            item["bounding_box"]["ymax"] = box[3]
        else:
            pass
    if debug:
        cv2.imwrite(os.path.join('data', "show.jpg"), image_draw)
    # regions1 = re_json['regions']
    # for ele in regions1:
    #     cv2.rectangle(image, (ele["bounding_box"]["xmin"], ele["bounding_box"]["ymin"]),
    #                   (ele["bounding_box"]["xmax"], ele["bounding_box"]["ymax"]), (0, 255, 0), 2)
    # cv2.imwrite(os.path.join('data', "show.jpg"), image)
    return re_json


if __name__ == '__main__':
    # '''服务端传入数据为json内数据 和图像
    # 使用方法：
    # new_json = adjust_item_edge(img_path, key_json)
    # key_json : regions 数组
    # new_json : 调整后的结果 size == key_json.size
    # '''
    #
    # print("前置解析")
    # file_dir = r"E:\data\aug_img\adjust"
    # img_path = os.path.join(file_dir, "7642572.jpg")
    # json_path = os.path.join(file_dir, "7642572.json")
    # print(img_path, json_path)
    # # 读取json
    # output_ios = open(json_path).read()
    # output_json = json.loads(output_ios)
    # for item in output_json:
    #     # print(item,output_json[item])
    #     if (item == "regions"):
    #         key_json = output_json[item]
    # # print(len(key_json))
    # for idx, item in enumerate(key_json):
    #     # print(key_json[idx])
    #     if (item["class_name"] == "alarm_info"):
    #         key_json[idx]["bounding_box"]["ymin"] -= 10
    #         key_json[idx]["bounding_box"]["ymax"] += 10
    #     # print(key_json[idx])
    #
    # new_json = adjust_item_edge(img_path, key_json)
    # for idx, val in enumerate(key_json):
    #     print(key_json[idx])
    #     print(new_json[idx])

    img_path = 'data/1.jpg'
    json_path = 'data/1.json'
    import ast
    f_read = open(json_path, 'r', encoding='utf-8').read()
    json_content = ast.literal_eval(f_read)
    img = cv2.imread(img_path)
    adjust_item_edge_by_gray_image(img, json_content)