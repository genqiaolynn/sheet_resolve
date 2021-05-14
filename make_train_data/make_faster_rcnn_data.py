# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2020/7/8 14:45'


import cv2, shutil, heapq
import xml.etree.cElementTree as ET
import utils
from PIL import Image, ImageEnhance
import shutil
import numpy as np
import os
import glob
import random


'''
制作faster rcnn的训练数据
功能：
1. 将原始图像进行图像增强
2. 合并需要合并的label
3. 检查object的宽高等信息
4. 然后重命名成六位格式的
5. 最后分成训练，验证等txt文件
'''


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    regions_dict = {}
    regions_bbox = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bbox_dict = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        region = {'class_name': class_name, 'bounding_box': bbox_dict}
        regions_bbox.append(region)
    bbox_dict = {'xml_name': xml_path, 'regions': regions_dict}
    return regions_bbox


def get_img_region_box2(im, bbox, label_size):
    height, width = im.shape[0], im.shape[1]
    height_radio = int(height) / float(label_size[1])
    width_radio = int(width) / float(label_size[0])
    xmin = int(float(bbox['xmin']) // float(width_radio))
    ymin = int(float(bbox['ymin']) // float(height_radio))
    xmax = int(float(bbox['xmax']) // float(width_radio))
    ymax = int(float(bbox['ymax']) // float(height_radio))
    # xmin = int(bbox['xmin'])
    # ymin = int(bbox['ymin'])
    # xmax = int(bbox['xmax'])
    # ymax = int(bbox['ymax'])
    region = [xmin, ymin, xmax, ymax]
    # region = im[ymin:ymax, xmin:xmax]
    return region


def get_img_region(img, bbox, label_size):
    height, width, _ = img.shape
    if height > width:
        h_ratio = int(height) / max(label_size)
        w_ratio = int(width) / min(label_size)
    else:
        h_ratio = int(height) / min(label_size)
        w_ratio = int(width) / max(label_size)
    xmin = int(float(bbox['xmin']) // w_ratio)
    ymin = int(float(bbox['ymin']) // h_ratio)
    xmax = int(float(bbox['xmax']) // w_ratio)
    ymax = int(float(bbox['ymax']) // h_ratio)
    region = [xmin, ymin, xmax, ymax]
    return region


def read_xml_and_resize_bbox_by_ratio1(raw_img_path, label_size):     # width:2000 height:1500
    rfind_path = raw_img_path[0: raw_img_path.rfind('\\')]
    resize_save_path = os.path.join(rfind_path, 'resize')
    if not os.path.exists(resize_save_path):
        os.makedirs(resize_save_path)
    xml_name_list = os.listdir(raw_img_path)
    # label_size = (3307, 2340)
    for xml_name in xml_name_list:
        if xml_name.endswith('.xml'):
            xml_path0 = os.path.join(raw_img_path, xml_name)
            print(xml_path0)
            regions_bbox = read_xml(xml_path0)
            img_path0 = os.path.join(raw_img_path, xml_name.replace('.xml', '.jpg'))

            # im = cv2.imread(img_path0)
            im1 = Image.open(img_path0)
            if im1.mode == 'RGBA':
                r, g, b, a = im1.split()
                im = Image.merge('RGB', (r, g, b))
            if im1.mode != 'RGB':
                im = im1.convert('RGB')
            else:
                im = im1
            im = np.array(im)
            tree = ET.parse(r'../exam_info/000000-template.xml')  # xml tree
            for index, region_info in enumerate(regions_bbox):
                bbox = region_info['bounding_box']
                im_region_list = get_img_region_box2(im, bbox, label_size)
                tree = utils.create_xml(region_info['class_name'], tree, im_region_list[0], im_region_list[1], im_region_list[2], im_region_list[3])

            save_path0 = os.path.join(resize_save_path, xml_name)
            tree.write(save_path0)
            image = cv2.resize(im, label_size, interpolation=cv2.INTER_AREA)
            # save_img = utils.write_single_img(image, os.path.join(save_path, xml_name.replace('.xml', '.jpg')))
            image_pil = Image.fromarray(image.astype('uint8')).convert('RGB')
            image_pil.save(os.path.join(resize_save_path, xml_name.replace('.xml', '.jpg')))
            resize_img_xml = os.path.join(resize_save_path, xml_name.replace('.xml', '.jpg'))
    return resize_save_path


def read_xml_and_resize_bbox_by_ratio2(xml_path, label_size):    # 根据长短边resize
    rfind_path = raw_img_path[0: raw_img_path.rfind('\\')]
    resize_save_path = os.path.join(rfind_path, 'resize')
    if not os.path.exists(resize_save_path):
        os.makedirs(resize_save_path)
    xml_list = glob.glob(xml_path + '\\*.xml')
    for xml_s in xml_list:
        regions = utils.read_xml(xml_s)
        img_path1 = xml_s.replace('.xml', '.jpg')
        # img = cv2.imread(img_path1)
        img = utils.read_single_img(img_path1)
        height, width = img.shape[0], img.shape[1]
        tree = ET.parse(r'../exam_info/000000-template.xml')  # xml tree
        for index, region_info in enumerate(regions):
            bbox = region_info['bounding_box']
            im_region_list = get_img_region(img, bbox, label_size)
            tree = utils.create_xml(region_info['class_name'], tree, im_region_list[0], im_region_list[1],
                                    im_region_list[2], im_region_list[3])
        xml_name = xml_s.split('\\')[-1]
        save_path0 = os.path.join(resize_save_path, xml_name)
        tree.write(save_path0)
        height, width, _ = img.shape
        if height > width:
            image = cv2.resize(img, (min(label_size), max(label_size)), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(img, (max(label_size), min(label_size)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(save_path0.replace('.xml', '.jpg'), image)
        utils.write_single_img(image, save_path0.replace('.xml', '.jpg'))
    return resize_save_path


def image_enhance(resize_img_xml):
    rfind_path = raw_img_path[0: raw_img_path.rfind('\\')]
    en_save_path = os.path.join(rfind_path, 'en')
    if not os.path.exists(en_save_path):
        os.makedirs(en_save_path)

    img_list = os.listdir(resize_img_xml)
    for img0 in img_list:
        if img0.endswith('.jpg'):
            img_name = img0.replace('.jpg', '')
            img_path0 = os.path.join(resize_img_xml, img0)

            img1 = Image.open(img_path0)
            if img1.mode == 'RGBA':
                r, g, b, a = img1.split()
                image = Image.merge('RGB', (r, g, b))
            if img1.mode != 'RGB':
                image = img1.convert('RGB')
            else:
                image = img1
            img = Image.fromarray(np.uint8(image))
            xml_path0 = os.path.join(resize_img_xml, img0.replace('.jpg', '.xml'))
            shutil.copyfile(img_path0, os.path.join(en_save_path, img0))
            shutil.copyfile(xml_path0, os.path.join(en_save_path, img0.replace('.jpg', '.xml')))

            # color
            color_ratio = [0.5, 1.5, 1.7, 2.0]
            for index, ratio in enumerate(color_ratio):
                im_1 = ImageEnhance.Color(img).enhance(ratio)
                save_name = img_name + '_' + str(index) + '_' + 'color' + '.jpg'
                img_save_path = os.path.join(en_save_path, save_name)
                im_1.save(img_save_path)
                xml_save_path = os.path.join(en_save_path, save_name.replace('.jpg', '.xml'))
                shutil.copyfile(os.path.join(resize_img_xml, img0.replace('.jpg', '.xml')), xml_save_path)
            # Brightness
            bright_ratio = [0.5, 0.7, 0.8, 0.9]
            for index, ratio in enumerate(bright_ratio):
                im_1 = ImageEnhance.Brightness(img).enhance(ratio)
                save_name = img_name + '_' + str(index) + '_' + 'bright' + '.jpg'
                img_save_path = os.path.join(en_save_path, save_name)
                im_1.save(img_save_path)
                xml_save_path = os.path.join(en_save_path, save_name.replace('.jpg', '.xml'))
                shutil.copyfile(os.path.join(resize_img_xml, img0.replace('.jpg', '.xml')), xml_save_path)
            # Contrast
            contrast_ratio = [0.5, 0.7, 0.8, 0.9]
            for index, ratio in enumerate(contrast_ratio):
                im_1 = ImageEnhance.Contrast(img).enhance(ratio)
                save_name = img_name + '_' + str(index) + '_' + 'contrast' + '.jpg'
                img_save_path = os.path.join(en_save_path, save_name)
                im_1.save(img_save_path)
                xml_save_path = os.path.join(en_save_path, save_name.replace('.jpg', '.xml'))
                shutil.copyfile(os.path.join(resize_img_xml, img0.replace('.jpg', '.xml')), xml_save_path)
            # Sharpness
            sharp_ratio = [0.5, 2.0, 2.5, 3.0]
            for index, ratio in enumerate(sharp_ratio):
                im_1 = ImageEnhance.Sharpness(img).enhance(ratio)
                save_name = img_name + '_' + str(index) + '_' + 'sharp' + '.jpg'
                img_save_path = os.path.join(en_save_path, save_name)
                im_1.save(img_save_path)
                xml_save_path = os.path.join(en_save_path, save_name.replace('.jpg', '.xml'))
                shutil.copyfile(os.path.join(resize_img_xml, img0.replace('.jpg', '.xml')), xml_save_path)
                print(xml_save_path)
    print('ok')
    return en_save_path


def get_training_file(jpeg_path, annotations_path, imagesets_path, all_size, test_size, train_size):
    jpg_list = glob.glob(jpeg_path + '\\*.jpg')
    xml_list = glob.glob(annotations_path + '\\*.xml')

    jpg_index = []
    for root, dirs, files in os.walk(jpeg_path):
        for file in files:
            index = file.replace('.jpg', '')
            jpg_index.append(index)

    xml_index = []
    for root, dirs, files in os.walk(annotations_path):
        for file in files:
            index = file.replace('.xml', '')
            xml_index.append(index)

    if len(jpg_index) > len(xml_index):
        print('the index of JPEGImages is more than that of  Annotations.')

    if len(jpg_index) < all_size:
        print('there is not enough img: ', len(jpg_index), 'while size needed  is ', all_size)

    if len(xml_index) >= all_size and len(jpg_index) >= all_size:
        test_list = random.sample(xml_index, test_size)
        trainval_list = list(set(xml_index).difference(set(test_list)))

        test_list = [i+'\n' for i in test_list]
        trainval_list = [i+'\n' for i in trainval_list]

        train_list = random.sample(trainval_list, train_size)
        # train_list = [i+'\n' for i in train_list]

        val_list = list(set(trainval_list).difference(set(train_list)))
        # val_list = [i+'\n' for i in val_list]

        with open(imagesets_path + '\\test.txt', 'w') as test_writter:
            # test_writter.writelines(sorted(test_list))
            test_writter.writelines(test_list)
        with open(imagesets_path + '\\train.txt', 'w') as train_writter:
            # train_writter.writelines(sorted(train_list))
            train_writter.writelines(train_list)
        with open(imagesets_path + '\\val.txt', 'w') as val_writter:
            # val_writter.writelines(sorted(val_list))
            val_writter.writelines(val_list)
        with open(imagesets_path + '\\trainval.txt', 'w') as trainval_writter:
            # trainval_writter.writelines(sorted(trainval_list))
            trainval_writter.writelines(trainval_list)


def combine_classes(en_save_path):
    xml_list = os.listdir(en_save_path)
    classes = ['class_w1', 'exam_number_w1', 'name_w', 'name_w1', 'room_w',
               'room_w1', 'school_w1', 'seat_w', 'seat_w1', 'student_info_w1']
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
            # img_path = xml_path1.replace('.xml', '.jpg')
            # shutil.copyfile(img_path, os.path.join(save_path, xml.replace('.xml', '.jpg')))

    # check box height width
    c_new_xml_path = os.listdir(new_xml_path)
    for c_xml in c_new_xml_path:
        c_xml_path = os.path.join(new_xml_path, c_xml)
        c_img_path = os.path.join(en_save_path, c_xml.replace('.xml', '.jpg'))
        # c_img = cv2.imread(c_img_path)
        c_img = utils.read_single_img(c_img_path)
        height1, width1 = c_img.shape[0], c_img.shape[1]

        tree = ET.parse(c_xml_path)
        root = tree.getroot()
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

            if int(xmax_old_text) > width1:
                new_xmax = str(width1)
                xmax.text = new_xmax

            if int(ymax_old_text) > height1:
                new_ymax = str(height1)
                ymax.text = new_ymax
        tree.write(c_xml_path)
    print('---check xml with height width done---')

    # delete xmin=xmax, ymin=ymax
    cc_new_xml_path = os.listdir(new_xml_path)
    for cc_xml in cc_new_xml_path:
        cc_xml_path = os.path.join(new_xml_path, cc_xml)
        cc_img_path = os.path.join(en_save_path, cc_xml.replace('.xml', '.jpg'))

        tree = ET.parse(cc_xml_path)
        root = tree.getroot()
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

            if int(xmin_old_text) >= int(xmax_old_text) or int(ymin_old_text) >= int(ymax_old_text):
                root.remove(obj)
        tree.write(cc_xml_path)

    save_path_ = ''
    for xml in xml_list:
        if xml.endswith('.xml'):
            xml_path2 = os.path.join(en_save_path, xml)
            save_path_ = os.path.join(en_save_path, 'raw_xml')
            if not os.path.exists(save_path_):
                os.makedirs(save_path_)
            shutil.move(xml_path2, os.path.join(save_path_, xml))
    try:
        if os.path.exists(save_path_):
            shutil.rmtree(save_path_)
        print('normal operate raw xml')
    except Exception as e:
        print('------check your raw xml------')

    save_xml_list = os.listdir(new_xml_path)
    for xml in save_xml_list:
        xml_path1 = os.path.join(new_xml_path, xml)
        final_xml_path = os.path.join(en_save_path, xml)
        shutil.move(xml_path1, final_xml_path)
    try:
        shutil.rmtree(new_xml_path)
        print('--------delete save xml---------')
    except Exception as e:
        print('----------somethong wrong with your save xml----------')

    return en_save_path


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
                f.write(item + ',' + '\n')
            elif index == len(region_list) - 1:
                f.write(item + '\n')
    print('deal with label is done')
    return region_list


def remame_xml_and_img_extreme0(raw_img_path):
    # resize_save_path = read_xml_and_resize_bbox_by_ratio1(raw_img_path, label_size)
    # resize_save_path = read_xml_and_resize_bbox_by_ratio2(raw_img_path, label_size)
    en_save_path = image_enhance(raw_img_path)
    combine_classes_path = combine_classes(en_save_path)

    region_list = crop_img_by_xmls_with_all_object2(combine_classes_path)

    rfind_path = raw_img_path[0: raw_img_path.rfind('\\')]
    save_path = os.path.join(rfind_path, 'rename')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path_jpg = os.path.join(save_path, 'JPEGImages')
    if not os.path.exists(save_path_jpg):
        os.makedirs(save_path_jpg)

    save_path_xml = os.path.join(save_path, 'Annotations')
    if not os.path.exists(save_path_xml):
        os.makedirs(save_path_xml)

    ImageSets_path = os.path.join(save_path, 'ImageSets')
    if not os.path.exists(ImageSets_path):
        os.makedirs(ImageSets_path)

    Main_path = os.path.join(ImageSets_path, 'Main')
    if not os.path.exists(Main_path):
        os.makedirs(Main_path)

    index = 1
    for root_dir, dirs, files in os.walk(combine_classes_path):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root_dir, file)
                img_name_new = '{:06d}'.format(index)
                img_path_new = os.path.join(save_path_jpg, img_name_new + '.jpg')
                shutil.copyfile(img_path, img_path_new)
                print(img_path_new)

                xml_path0 = img_path.replace('.jpg', '.xml')
                new_xml_path = os.path.join(save_path_xml, img_name_new + '.xml')

                shutil.copyfile(xml_path0, new_xml_path)
                index += 1

                print(new_xml_path)

    size = len(glob.glob(save_path_jpg + '\\*.jpg'))
    test_size = 50
    val_size = int(size * 0.2)
    train_size = size - test_size - val_size
    get_training_file(save_path_jpg, save_path_xml, Main_path, size, test_size, train_size)
    shutil.rmtree(en_save_path)
    print('-----------delete enhance folder-------------')
    print('-----successfully-----')


if __name__ == '__main__':
    raw_img_path = r'E:\December\physic\combine_all\mix\test\1'
    remame_xml_and_img_extreme0(raw_img_path)

