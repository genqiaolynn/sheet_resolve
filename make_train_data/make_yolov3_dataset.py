# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/1/13 17:32'


'''
1. 修改cfg中的anchor，这个先验框是根据不同数据集GT box的宽高聚类出来的
2. 修改cfg中yolo层上面那个conv核大小，公式为3*(classes + 5)
3. 制作数据
'''

import re, os, glob, random, shutil
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

# subject_id = [
# 'math',
# 'english',
# 'chinese',
# 'physics',
# 'chemistry',
# 'biology',
# 'politics',
# 'history',
# 'geography',
# 'science_comprehensive',
# 'arts_comprehensive',
# 'math_blank',
# 'chinese_blank',
# 'science_comprehensive_blank',
# 'arts_comprehensive_blank',
# ]

img_format = ['.jpg', '.jpeg', '.tif', '.bmp', '.png']
# classes = open('./data/math_blank/VOC2007/label.txt', 'r', encoding='utf-8').read().splitlines()[0].split(',')


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def get_test_data(img_list, test_index):
    test_save_path = os.path.join(save_path, 'test')
    mkdir(test_save_path)

    for ele in test_index:
        img_name = img_list[ele].split('\\')[-1]
        shutil.copyfile(img_list[ele], os.path.join(test_save_path, img_name))


def convert(size, bbox):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return [x, y, w, h]


def get_train_label(img_list, all_labels_path, classes):
    for img_path in img_list:
        xml_path = img_path.replace('.jpg', '.xml')
        xml_name = xml_path.split('\\')[-1].replace('.xml', '')
        img_raw = ''
        try:
            img_raw = Image.open(img_path).convert("RGB")  # 1
        except:
            pass
        img_raw1 = np.array(img_raw)
        w = img_raw1.shape[: 2][1]
        h = img_raw1.shape[: 2][0]

        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bbox = convert((w, h), [xmin, ymin, xmax, ymax])
            cls_id = classes.index(name)
            with open(os.path.join(all_labels_path, xml_name + '.txt').replace('\\', '/'), 'a', encoding='utf-8') as ff:
                ff.write(str(cls_id) + ' ' + ' '.join([str(box) for box in bbox]) + '\n')


def get_all_labels(img_list):
    classes = []
    for img_path in img_list:
        xml_path = img_path.replace('.jpg', '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            classes.append(name)
    classes_ = sorted(list(set(classes)))
    return classes_


def get_yolov3_dataset(img_path, save_path, test_percent, val_percent):
    img_name_list = os.listdir(img_path)
    img_list = [os.path.join(img_path, ele) for ele in img_name_list if os.path.splitext(ele)[-1].lower() in img_format]
    classes = get_all_labels(img_list)

    # all_labels
    all_labels_path = os.path.join(save_path, 'labels')
    mkdir(all_labels_path)

    get_train_label(img_list, all_labels_path, classes)   # 将voc转成yolo格式

    # all_images
    all_img_path = os.path.join(save_path, 'images')
    mkdir(all_img_path)
    for ele in img_list:
        img_name = ele.split('\\')[-1]
        shutil.copyfile(ele, os.path.join(all_img_path, img_name))
    new_all_img_list = glob.glob(all_img_path + '\\*.jpg')

    # test data
    test_index = random.sample([i for i in range(len(img_list))], int(len(img_list)*test_percent))
    get_test_data(img_list, test_index)

    trainval_index = list(set([i for i in range(len(img_list))]).difference(set(test_index)))
    # get train data and val data
    val_index = random.sample(trainval_index, int(len(trainval_index) * val_percent))
    train_index = list(set(trainval_index).difference(set(val_index)))
    for t_index in train_index:
        train_txt_path = os.path.join(save_path, 'train.txt').replace('\\', '/')
        with open(train_txt_path, 'a', encoding='utf-8') as ftrain:
            ftrain.write(new_all_img_list[t_index].replace('\\', '/') + '\n')

    for v_index in val_index:
        val_txt_path = os.path.join(save_path, 'test.txt').replace('\\', '/')
        with open(val_txt_path, 'a', encoding='utf-8') as fval:
            fval.write(new_all_img_list[v_index].replace('\\', '/') + '\n')

    # class.names
    for index, ele in enumerate(classes):
        class_names_path = os.path.join(save_path, 'class.names').replace('\\', '/')
        with open(class_names_path, 'a', encoding='utf-8') as f2:
            if index == len(classes)-1:
                f2.write(ele)
            else:
                f2.write(ele + '\n')
    # class.data
    with open(os.path.join(save_path, 'class.data'), 'a', encoding='utf-8') as f3:
        f3.write('classes=' + str(len(classes)) + '\n')
        f3.write('train=' + train_txt_path + '\n')
        f3.write('valid=' + val_txt_path + '\n')
        f3.write('names=' + class_names_path + '\n')
    print('done')


if __name__ == '__main__':
    img_path = r'E:\December\math_12_18\1_18\blank\en'
    save_path = r'E:\December\math_12_18\1_18\blank\blank'
    test_percent = 0.02
    val_percent = 0.2
    get_yolov3_dataset(img_path, save_path, test_percent, val_percent)
