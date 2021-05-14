# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/24 15:29'


import os, glob, utils, cv2, shutil
from utils import create_xml
import xml.etree.ElementTree as ET


def get_new_exam_number_s(img_path):
    xml_list = glob.glob(img_path + '\\*.xml')
    for xml in xml_list:
        img_path1 = xml.replace('.xml', '.jpg')
        sheet_dict = utils.read_xml_to_json(xml)
        xml_name = xml.split('\\')[-1]
        exam_number_s = [ele for ele in sheet_dict['regions'] if ele['class_name'] == 'exam_number_s']
        exam_number = [ele for ele in sheet_dict['regions'] if ele['class_name'] == 'exam_number']
        regions = [ele for ele in sheet_dict['regions'] if ele['class_name'] != 'exam_number_s']
        new_exam_number_s = []
        try:
            if len(exam_number) == 1:
                exam_number_s = sorted(exam_number_s, key=lambda k: k['bounding_box']['xmin'])
                all_index = [i for i in range(len(exam_number_s))]
                get_index = all_index[::2]
                new_exam_number_s = [exam_number_s[ele] for ele in get_index]
        except Exception as E:
            print(xml)

        new_regions = regions + new_exam_number_s
        # choice_s的操作
        try:
            choice_s = [ele for ele in new_regions if ele['class_name'] == 'choice_s']
            choice_m = [ele for ele in new_regions if ele['class_name'] == 'choice_m']
            new_regions_tmp = [ele for ele in new_regions if ele['class_name'] != 'choice_m' and ele['class_name'] != 'choice_s']
            choice_s_and_choice_m_list = []
            for ele_m in choice_m:
                loc_m = [ele_m['bounding_box']['xmin'], ele_m['bounding_box']['ymin'],
                         ele_m['bounding_box']['xmax'], ele_m['bounding_box']['ymax']]
                choice_s_list_tmp = []
                for ele_s in choice_s:
                    loc_s = [ele_s['bounding_box']['xmin'], ele_s['bounding_box']['ymin'],
                             ele_s['bounding_box']['xmax'], ele_s['bounding_box']['ymax']]
                    if utils.decide_coordinate_contains1(loc_s, loc_m) == True:   # 后面的框是大框
                        choice_s_list_tmp.append(ele_s)
                if len(choice_s_list_tmp) == 1:
                    choice_s_and_choice_m_list.append(ele_m)
                    break
                else:
                    all_choice_s_index = [i for i in range(len(choice_s_list_tmp))]
                    get_choice_s_index = all_choice_s_index[::2]
                    choice_s_list = [choice_s_list_tmp[ele] for ele in get_choice_s_index]
                    choice_s_and_choice_m_list.append(ele_m)
                    choice_s_and_choice_m_list += choice_s_list
            all_new_regions = choice_s_and_choice_m_list + new_regions_tmp
            xml_template = r'F:\exam\sheet_resolve\exam_info\000000-template.xml'
            tree = ET.parse(xml_template)
            for ele in all_new_regions:
                name = ele['class_name']
                xmin = ele['bounding_box']['xmin']
                ymin = ele['bounding_box']['ymin']
                xmax = ele['bounding_box']['xmax']
                ymax = ele['bounding_box']['ymax']
                create_xml(name, tree, xmin, ymin, xmax, ymax)
            tree.write(os.path.join(r'E:\December\math_12_18\1_18\modify_data1\new', xml_name))
            shutil.copyfile(img_path1, os.path.join(r'E:\December\math_12_18\1_18\modify_data1\new', xml_name.replace('.xml', '.jpg')))
        except Exception as E:
            print(xml)


if __name__ == '__main__':
    img_path = r'E:\December\math_12_18\1_18\modify_data1\combine_class'
    get_new_exam_number_s(img_path)
