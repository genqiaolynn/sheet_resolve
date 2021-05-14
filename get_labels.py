# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/4/15 18:43'


import glob, os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


'''
拿到所有xml的label
'''


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


if __name__ == '__main__':
    xml_path = r'E:\December\chinese\blank\raw'
    crop_img_by_xmls_with_all_object2(xml_path)