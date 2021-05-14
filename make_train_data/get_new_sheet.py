# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/24 14:44'


import glob, os, utils, shutil
import xml.etree.ElementTree as ET


def val_class():
    xml_path = r'E:\December\math_12_18\1_18\blank\en'
    xml_list = glob.glob(xml_path + '\\*.xml')

    for xml in xml_list:
        if xml.endswith('.xml'):
            tree = ET.parse(xml)
            root = tree.getroot()
            # remove old qr_code
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'w_h_blank' or name == 'line_0' or name == 'line_1' or\
                        name == 'line_2' or name == 'line_3' or name == 'print_info':
                    root.remove(obj)
            tree.write(xml)
            print(xml)


def val_class_():
    xml_path = r'E:\December\math_12_18\1_18\blank\en'
    xml_list = glob.glob(xml_path + '\\*.xml')

    for xml in xml_list:
        if xml.endswith('.xml'):
            tree = ET.parse(xml)
            root = tree.getroot()
            # remove old qr_code
            for obj in root.findall('object'):
                name = obj.find('name').text
                class_name = ['w_h_blank', 'line_0', 'line_1', 'line_2', 'class_w1', 'exam_number_w1',
                              'name_w', 'name_w1', 'room_w', 'room_w1', 'verify', 'school_w1', 'seat_w',
                              'seat_w1', 'student_info_w1', 'executor', 'total_score', 'time', 'student_info_w', 'type']
                if name in class_name:
                    root.remove(obj)
            tree.write(xml)
            print(xml)


def combine_classes(en_save_path):
    xml_list = os.listdir(en_save_path)
    classes = ['class_w1', 'exam_number_w1', 'name_w', 'name_w1', 'room_w', 'room_w1',
               'verify', 'school_w1', 'seat_w', 'seat_w1', 'student_info_w1', 'executor',
               'total_score', 'time', 'total_score']
    new_xml_path = ''
    for xml in xml_list:
        if xml.endswith('.xml'):
            xml_path1 = os.path.join(en_save_path, xml)
            tree = ET.parse(xml_path1)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name')
                if name.text in classes:
                    obj.find('name').text = name.text.replace(name.text, 'student_info')
            new_xml_path = os.path.join(en_save_path, 'save')
            if not os.path.exists(new_xml_path):
                os.makedirs(new_xml_path)
            tree.write(os.path.join(new_xml_path, xml))
            print(os.path.join(new_xml_path, xml))
            # img_path = xml_path1.replace('.xml', '.jpg')
            # shutil.copyfile(img_path, os.path.join(save_path, xml.replace('.xml', '.jpg')))

    # check box height width
    c_new_xml_path = os.listdir(new_xml_path)
    for c_xml in c_new_xml_path:
        c_xml_path = os.path.join(new_xml_path, c_xml)
        # c_img_path = os.path.join(en_save_path, c_xml.replace('.xml', '.jpg'))
        # c_img = cv2.imread(c_img_path)
        # c_img = utils.read_single_img(c_img_path)
        # height1, width1 = c_img.shape[0], c_img.shape[1]

        tree = ET.parse(c_xml_path)
        root = tree.getroot()
        size = root.findall('size')[0]
        width = size.findall('width')[0].text
        height = size.findall('height')[0].text
        for obj in root.findall('object'):

            bbox = obj.find('bndbox')
            xmin = bbox.find('xmin')
            xmin_old_text = xmin.text

            ymin = bbox.find('ymin')
            ymin_old_text = ymin.text

            xmax = bbox.find('xmax')
            xmax_old_text = xmax.text

            ymax = bbox.find('ymax')
            ymax_old_text = ymax.text

            if int(xmin_old_text) < 0:
                new_xmin = str(0)
                xmin.text = new_xmin

            if int(ymin_old_text) < 0:
                new_ymin = str(0)
                ymin.text = new_ymin

            if int(xmax_old_text) > int(width):
                new_xmax = width
                xmax.text = new_xmax

            if int(ymax_old_text) > int(height):
                new_ymax = height
                ymax.text = new_ymax
        tree.write(c_xml_path)
    print('---check xml with height width done---')


def crop_img_by_xmls_with_all_object2(xml_path):
    xml_list = glob.glob(xml_path + '\\*.xml')
    region_lists = []
    for xml in xml_list:
        if xml.endswith('.xml'):
            tree = ET.parse(xml)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                region_lists.append(name)
    region_set = list(set(region_lists))
    region_list = sorted(region_set, key=str.lower)

    save_path = xml_path[0: xml_path.rfind('\\')]
    save_path0 = os.path.join(save_path, 'label.txt')
    with open(save_path0, 'a', encoding='utf-8') as f:
        for index, item in enumerate(region_list):
            if index < len(region_list) - 1:
                f.write(item + ',')
            elif index == len(region_list) - 1:
                f.write(item)
    print(region_list)
    print('length_of_label:', len(region_list))


def get_labels(xml_path):
    xml_list = glob.glob(xml_path + '\\*.xml')
    save_path = r'E:\December\math_12_18\1_18\modify_data1\save'
    region_lists = []
    for xml in xml_list:
        xml_name = xml.split('\\')[-1]
        if xml.endswith('.xml'):
            tree = ET.parse(xml)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'cloze_score' or name == 'print_info':
                    shutil.copyfile(xml, os.path.join(save_path, xml_name))
                    shutil.copyfile(xml.replace('.xml', '.jpg'), os.path.join(save_path, xml_name.replace('.xml', '.jpg')))


if __name__ == '__main__':
    # img_path = r'E:\December\math_12_18\1_18\modify_data\raw'
    # check_label_region(img_path)

    # en_save_path = r'E:\December\math_12_18\1_18\unblank\new\Annotations'
    # combine_classes(en_save_path)

    # val_class()
    # val_class_()


    xml_path = r'E:\December\math_12_18\1_18\blank\en'
    crop_img_by_xmls_with_all_object2(xml_path)

    # xml_path = r'E:\December\math_12_18\1_18\blank\en'
    # get_labels(xml_path)