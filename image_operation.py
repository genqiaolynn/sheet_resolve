# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/1/29 15:52'


from PIL import Image
import numpy as np
import cv2, os, shutil, re, glob
import xml.etree.ElementTree as ET
from utils import *


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
                if name == 'w_h_blank' or name == 'line_0' or name == 'line_1' \
                        or name == 'line_2' or name == 'solve_without_type_score' or name == 'line_3' or name == 'solve_with_type_score_without_aspect':
                    root.remove(obj)
            tree.write(xml)
            print(xml)


def convert_img_mode():
    img_path = r'E:\7_2_test_img\00_x线上测试结果tmp_file_1_29\math\00000_908aa6ede5.jpg'
    raw_img = Image.open(img_path)
    if raw_img.mode == 'P':
        img = raw_img.convert('RGB')
    elif raw_img.mode == 'L':   # 二值图
        channel = raw_img.split()
        img = Image.merge('RGB', (channel[0], channel[1], channel[2]))
    elif raw_img.mode == 'RGBA':
        img = Image.new('RGB', raw_img.size, (255, 255, 255))
        img.paste(raw_img, mask=raw_img.split()[3])   # 3是透明的那个通道
    else:
        img = raw_img
    opencv_img = np.array(img)
    cv2.imwrite('1.jpg', opencv_img)


def move_small_xml_jpg():
    img_path = r'E:\sabe\1'
    save_path = r'E:\sabe\2'
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if file.endswith('.xml'):
                # name = file.split('_')[-1].replace('.jpg', '')
                name_list = file.split('_')
                small_ = [ele for ele in name_list if re.findall('small', ele)]
                if len(small_) > 0:
                    continue
                else:
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(save_path, file)
                    shutil.move(old_path, new_path)

                    old_path1 = os.path.join(root, file.replace('.xml', '.jpg'))
                    new_path1 = os.path.join(save_path, file.replace('.xml', '.jpg'))
                    shutil.move(old_path1, new_path1)


def get_raw_xml():
    img_path = r'E:\sabe\208\raw'
    save_path = r'E:\sabe\208\save'
    img_list = glob.glob(img_path + '\\*.jpg')
    for ele in img_list:
        name = ele.split('\\')[-1]
        small = re.findall('raw', name)
        if len(small) != 0:
            shutil.move(ele, os.path.join(save_path, name))
            shutil.move(ele.replace('.jpg', '.xml'), os.path.join(save_path, name.replace('.jpg', '.xml')))
            print(ele)


if __name__ == '__main__':
    # convert_img_mode()

    val_class()

    # get_raw_xml()