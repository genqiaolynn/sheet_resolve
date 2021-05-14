# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/1 19:15'


import ast, glob
import numpy as np
from utils import *
from sheet_adjust import adjust_item_edge_by_gray_image


'''
后期再优化，可优化方向：只有中间那个一栏有框，只有左边有框，只有右边有框这些情况的处理

'''


def get_col_split_index_xywh(sheet_dict):
    regions = sheet_dict['regions']
    regions = [ele for ele in regions if re.findall('line', ele['class_name']) == []]

    # xyxy
    bbox_xyxy = [[ele['bounding_box']['xmin'], ele['bounding_box']['ymin'],
                  ele['bounding_box']['xmax'], ele['bounding_box']['ymax']] for ele in regions]

    bbox_xywh = xyxy2xywh(bbox_xyxy)
    bbox_xywh_list = list(bbox_xywh)
    bbox_xywh_list = [list(map(int, ele)) for ele in bbox_xywh_list]
    bbox_xywh_list = sorted(bbox_xywh_list, key=lambda k: k[0], reverse=True)

    bbox_xyxy_arr = np.array(bbox_xywh_list)
    rear = bbox_xyxy_arr[1:, 0]
    former = bbox_xyxy_arr[:-1, 0]
    differ = rear - former
    differ_list = list(differ)
    top_three_index = np.argsort(differ_list)[::-1][:3]
    three_box = [bbox_xywh_list[ele+1] for ele in top_three_index]
    three_box_xyxy = xywh2xyxy(three_box)
    print(three_box_xyxy)


def check_regions(regions_raw):
    regions = regions_raw.copy()
    solve_list = [ele for ele in regions if ele['class_name'] == 'solve' or ele['class_name'] == 'solve0']
    if len(solve_list) != 0:
        solve_bbox = [[ele['bounding_box']['xmin'], ele['bounding_box']['ymin'],
                      ele['bounding_box']['xmax'], ele['bounding_box']['ymax']] for ele in solve_list]
        solve_xywh = xyxy2xywh(solve_bbox)    # array --> format

        regions_bbox = [[ele['bounding_box']['xmin'], ele['bounding_box']['ymin'],
                      ele['bounding_box']['xmax'], ele['bounding_box']['ymax']] for ele in regions]
        regions_xywh = xyxy2xywh(regions_bbox)
        width_mean = int(np.mean(solve_xywh[:, 2]))
        delete_index = np.where(regions_xywh[:, 2] > int(1.2 * width_mean))[0]
        if delete_index:
            for i in delete_index:
                del regions[i]
        return regions
        # regions = [for ele in delete_index]

    else:
        regions = regions_raw
        return regions


def get_iou(bbox1, bbox2):
    # bbox1: 分割点的框
    # 区域框
    # iou = inter_area / union_area

    bbox1_array = np.array(bbox1)
    bbox2_array = np.array(bbox2)
    bbox1_array = bbox1_array[np.newaxis, :]

    xx1 = np.maximum(bbox1_array[:, 0], bbox2_array[:, 0])
    yy1 = np.maximum(bbox1_array[:, 1], bbox2_array[:, 1])
    xx2 = np.minimum(bbox1_array[:, 2], bbox2_array[:, 2])
    yy2 = np.minimum(bbox1_array[:, 3], bbox2_array[:, 3])

    inter_area = (xx2 - xx1) * (yy2 - yy1)
    print(inter_area)


def get_col_split_index_raw(sheet_dict, img_size, xml_path, img):
    sheet_dict = adjust_item_edge_by_gray_image(img, sheet_dict)
    regions = sheet_dict['regions']
    # 去掉异常区域
    regions = check_regions(regions)
    regions = [ele for ele in regions if re.findall('line|w_h_blank|solve_without_type_score', ele['class_name']) == []]
    # xyxy
    bbox_xyxy = [[ele['bounding_box']['xmin'], ele['bounding_box']['ymin'],
                  ele['bounding_box']['xmax'], ele['bounding_box']['ymax']] for ele in regions]

    bbox_xyxy = sorted(bbox_xyxy, key=lambda k: k[2])
    bbox_xyxy_arr = np.array(bbox_xyxy)
    rear = bbox_xyxy_arr[1:, 0]
    former = bbox_xyxy_arr[:-1, 0]
    # rear = bbox_xyxy_arr[1:, 3]
    # former = bbox_xyxy_arr[:-1, 1]
    differ = rear - former
    differ_list = list(differ)
    # above_zero = [abs(ele) for ele in list1]
    # above_zero = [ele for ele in differ_list if ele > 0]
    above_zero = np.argsort(differ_list)[::-1][:3]
    three_box = [bbox_xyxy_arr[ele] for ele in above_zero]
    suspect_index = sorted([ele[2]+13 for ele in three_box])    # 所有候选的分割点增加10个像素

    # abandon error
    inter_box = []
    for split_point in suspect_index:
        # split_box = [split_point, 1, split_point + 1, img_size[1]]
        point_list = []
        for ele in bbox_xyxy:
            if ele[0] <= split_point <= ele[2]:
                point_list.append(ele)
        sheet_nbs = {}
        sheet_nbs['point'] = split_point
        sheet_nbs['bbox'] = point_list
        inter_box.append(sheet_nbs)

    # inter_box_length = [len(ele['bbox']) for ele in inter_box if len(ele['bbox']) <= 5]
    inter_box_index = []
    delete_s = []
    for index, ele in enumerate(inter_box):
        if len(ele['bbox']) > 5:
            delete_s.append(ele)
        else:
            inter_box_index.append(index)
    suspect_index_new = [suspect_index[ele] for ele in inter_box_index]
    print(suspect_index_new)
    # 去掉可能存在的冗余

    suspect_index_new_array = np.array(suspect_index_new)
    former_ = suspect_index_new_array[:-1]
    rear_ = suspect_index_new_array[1:]
    differ_ = rear_ - former_
    suspect_index_new_ = []
    if len(differ_) > 0 and (differ_ > 260).all() == True:
        suspect_index_new_ = suspect_index_new
    elif len(differ_) == 2:
        index_less_ = np.where(differ_ < 250)[0]
        if len(index_less_) == 1 and index_less_[0] != 0:
            trouble_index = [inter_box_index[index_less_[0]], inter_box_index[index_less_[0]] + 1]
            normal_index = list(set(inter_box_index) - set(trouble_index))    # 这里只能确定正常的索引值，不正常的靠这种方式不能正确判断出来
            max_width_all_ = [ele[2] - ele[0] for ele in bbox_xyxy]
            for ele in trouble_index:
                if (max_width_all_ < suspect_index_new[ele]).all():
                    suspect_index_new_.append(suspect_index_new[ele])
                else:
                    continue
            suspect_index_new_.append(suspect_index_new[normal_index[0]])
        elif len(index_less_) == 1 and index_less_[0] == 0:
            trouble_index = [inter_box_index[index_less_[0]], inter_box_index[index_less_[0]] + 1]
            normal_index = list(set(inter_box_index) - set(trouble_index))
            suspect_index_new_ = [int(np.mean(np.array(suspect_index_new[trouble_index[0]],
                                                       suspect_index_new[trouble_index[1]]))), suspect_index_new[normal_index[0]]]

        else:
            # suspect_index_new_ = [int(np.mean((suspect_index_new[0], suspect_index_new[1])))]
            suspect_index_new_ = [max(suspect_index_new)]

    elif len(differ_) == 1:
        suspect_index_new_ = [int(np.mean(suspect_index_new))]

    max_width_all = [ele[2]-ele[0]-10 for ele in bbox_xyxy]
    max_width_all_array = np.array(max_width_all)
    suspect_index_new_ = sorted(suspect_index_new_)
    final_index = []
    for index, ele in enumerate(suspect_index_new_):
        if index == 0:
            # if len(np.where(max_width_all_array < ele)[0]) > 0:
            if (max_width_all_array < ele).all():
                final_index.append(ele)
            else:
                continue
        elif index == len(suspect_index_new_)-1:
            right_size = img_size[0] - ele
            if (max_width_all_array < right_size).all():
                final_index.append(ele)
            else:
                continue
        else:
            final_index.append(ele)

    # delete_index = []
    # for ele in inter_box:
    #     len_box = ele['bbox']
    #     if len(delete_s) == 0 and len(len_box) > int(np.mean(np.array(inter_box_length))) or len(ele) > 5:
    #         delete_index.append(ele['point'])
    #     elif len(delete_s) != 0 and len(inter_box_length) == 0 and len(len_box) > int(np.mean(np.array(inter_box_length))) or len(ele) > 5:
    #         delete_index.append(ele['point'])
    #     elif len(delete_s) != 0 and len(len_box) > int(np.mean(np.array(inter_box_length)))+1 or len(ele) > 5:
    #         delete_index.append(ele['point'])
    #     else:
    #         continue
    # if len(delete_index) > 0:
    #     for ele in delete_index:
    #         suspect_index.remove(ele)

    if len(final_index) != 0:
        template = r'F:\exam\sheet_resolve\exam_info\000000-template.xml'
        tree = ET.parse(template)
        for ele in final_index:
            create_xml('line', tree, int(ele), 1, int(ele)+1, img_size[1])
        name = xml_path.split('\\')[-1]
        tree.write(r'E:\sabe\208\save\\' + name)
        cv2.imwrite(r'E:\sabe\208\save\\' + name.replace('.xml', '.jpg'), img)


def get_col_split_index(sheet_dict, img_size, xml_path, img):
    # sheet_dict = adjust_item_edge_by_gray_image(img, sheet_dict)
    regions = sheet_dict['regions']
    # 去掉异常区域
    regions = check_regions(regions)
    regions = [ele for ele in regions if re.findall('line|w_h_blank|solve_without_type_score', ele['class_name']) == []]

    # xyxy
    bbox_xyxy = [[ele['bounding_box']['xmin'], ele['bounding_box']['ymin'],
                  ele['bounding_box']['xmax'], ele['bounding_box']['ymax']] for ele in regions]

    bbox_xyxy = sorted(bbox_xyxy, key=lambda k: k[2])
    bbox_xyxy_arr = np.array(bbox_xyxy)
    rear = bbox_xyxy_arr[1:, 0]
    former = bbox_xyxy_arr[:-1, 0]
    differ = rear - former
    differ_list = list(differ)
    # above_zero = [abs(ele) for ele in list1]
    # above_zero = [ele for ele in differ_list if ele > 0]
    above_zero = np.argsort(differ_list)[::-1][:3]
    three_box = [bbox_xyxy_arr[ele] for ele in above_zero]
    suspect_index = sorted([ele[2]+13 for ele in three_box])    # 所有候选的分割点增加10个像素

    # abandon error
    inter_box = []
    for split_point in suspect_index:
        # split_box = [split_point, 1, split_point + 1, img_size[1]]
        point_list = []
        for ele in bbox_xyxy:
            if ele[0] <= split_point <= ele[2]:
                point_list.append(ele)
        sheet_nbs = {}
        sheet_nbs['point'] = split_point
        sheet_nbs['bbox'] = point_list
        inter_box.append(sheet_nbs)

    # inter_box_length = [len(ele['bbox']) for ele in inter_box if len(ele['bbox']) <= 5]
    inter_box_index = []
    delete_s = []
    for index, ele in enumerate(inter_box):
        if len(ele['bbox']) >= 4:
            delete_s.append(ele)
        else:
            inter_box_index.append(index)
    suspect_index_new = [suspect_index[ele] for ele in inter_box_index]
    print(suspect_index_new)
    # 去掉可能存在的冗余

    suspect_index_new_array = np.array(suspect_index_new)
    former_ = suspect_index_new_array[:-1]
    rear_ = suspect_index_new_array[1:]
    differ_ = rear_ - former_
    suspect_index_new_ = []
    max_width_all_ = [ele[2] - ele[0] for ele in bbox_xyxy]
    if len(differ_) > 0 and (differ_ > 310).all() == True:
        if (max_width_all_ < suspect_index_new[0]).all():
            suspect_index_new_ = suspect_index_new
        else:
            suspect_index_new_ = suspect_index_new[1:]
    elif len(differ_) == 2:
        index_less_ = np.where(differ_ < 310)[0]
        if len(index_less_) == 1 and index_less_[0] != 0:
            trouble_index = [inter_box_index[index_less_[0]], inter_box_index[index_less_[0]] + 1]
            normal_index = list(set(inter_box_index) - set(trouble_index))    # 这里只能确定正常的索引值，不正常的靠这种方式不能正确判断出来
            for ele in trouble_index:
                if (max_width_all_ < suspect_index_new[ele]).all():
                    suspect_index_new_.append(suspect_index_new[ele])
                else:
                    continue
            suspect_index_new_.append(suspect_index_new[normal_index[0]])
        elif len(index_less_) == 1 and index_less_[0] == 0:
            trouble_index = [inter_box_index[index_less_[0]], inter_box_index[index_less_[0]] + 1]
            normal_index = list(set(inter_box_index) - set(trouble_index))
            suspect_index_new_ = [int(np.mean(np.array(suspect_index_new[trouble_index[0]],
                                                       suspect_index_new[trouble_index[1]]))), suspect_index_new[normal_index[0]]]

        else:
            # suspect_index_new_ = [int(np.mean((suspect_index_new[0], suspect_index_new[1])))]
            suspect_index_new_ = [max(suspect_index_new)]

    elif len(differ_) == 1:
        suspect_index_new_ = [int(np.mean(suspect_index_new))]

    max_width_all = [ele[2]-ele[0]-10 for ele in bbox_xyxy]
    max_width_all_array = np.array(max_width_all)
    suspect_index_new_ = sorted(suspect_index_new_)
    final_index = []
    for index, ele in enumerate(suspect_index_new_):
        if index == 0:
            # if len(np.where(max_width_all_array < ele)[0]) > 0:
            if (max_width_all_array < ele).all():
                final_index.append(ele)
            else:
                continue
        elif index == len(suspect_index_new_)-1:
            right_size = img_size[0] - ele
            if (max_width_all_array < right_size).all():
                final_index.append(ele)
            else:
                continue
        else:
            final_index.append(ele)

    # 如果两根线相差在一定阈值范围内，就判断，取跟框交点最小的那个线
    # xx = final_index
    # yy = inter_box

    # delete_index = []
    # for ele in inter_box:
    #     len_box = ele['bbox']
    #     if len(delete_s) == 0 and len(len_box) > int(np.mean(np.array(inter_box_length))) or len(ele) > 5:
    #         delete_index.append(ele['point'])
    #     elif len(delete_s) != 0 and len(inter_box_length) == 0 and len(len_box) > int(np.mean(np.array(inter_box_length))) or len(ele) > 5:
    #         delete_index.append(ele['point'])
    #     elif len(delete_s) != 0 and len(len_box) > int(np.mean(np.array(inter_box_length)))+1 or len(ele) > 5:
    #         delete_index.append(ele['point'])
    #     else:
    #         continue
    # if len(delete_index) > 0:
    #     for ele in delete_index:
    #         suspect_index.remove(ele)

    if len(final_index) != 0:
        template = r'F:\exam\sheet_resolve\exam_info\000000-template.xml'
        tree = ET.parse(template)
        for ele in final_index:
            create_xml('line', tree, int(ele), 1, int(ele)+1, img_size[1])
        name = xml_path.split('\\')[-1]
        tree.write(r'E:\sabe\208\save\\' + name)
        cv2.imwrite(r'E:\sabe\208\save\\' + name.replace('.xml', '.jpg'), img)


if __name__ == '__main__':
    # json_path = 'data/1.json'
    # ff = open(json_path, 'r', encoding='utf-8').read()
    # sheet_dict = ast.literal_eval(ff)

    xml_path = r'E:\sabe\208\raw_xml'
    # xml_path = r'E:\sabe\208\1'
    xml_list = glob.glob(xml_path + '\\*.xml')
    for xml in xml_list:
        sheet_dict = read_xml_to_json(xml)
        xml_path0 = sheet_dict['xml_path']
        img_path = xml_path0.replace('.xml', '.jpg')
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img_size = (w, h)
        get_col_split_index(sheet_dict, img_size, xml_path0, img)