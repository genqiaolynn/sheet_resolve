from __future__ import division
# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2020/11/24 17:47'


'''
根据GT宽高聚类用的是resize过的
'''

import xml.etree.cElementTree as ET
import numpy as np
import os, shutil
import sys, cv2, glob, torchvision
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class DataSet(data.Dataset):
    def __init__(self, img_dir, resize):
        super(DataSet, self).__init__()
        self.img_paths = [ele.replace('\\', '/') for ele in sorted(glob.glob('{:s}/*'.format(img_dir))) if ele.endswith('.jpg')]
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize))])

    def __getitem__(self, item):
        img_raw = Image.open(self.img_paths[item]).convert('RGB')
        img_transform = self.transform(img_raw)

        return img_raw, img_transform, self.img_paths[item]

    def __len__(self):
        return len(self.img_paths)


def read_xml_to_json(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    regions_list = []
    width = int(root.findall('size')[0].find('width').text)
    height = int(root.findall('size')[0].find('height').text)

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
    sheet_dict = {'regions': regions_list, 'img_size': (width, height)}
    return sheet_dict


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def parse_anno(annotation_path, target_size=None):
    anno = open(annotation_path, 'r')
    result = []
    for line in anno:
        s = line.strip().split(' ')
        img_w = int(s[2])
        img_h = int(s[3])
        s = s[4:]
        box_cnt = len(s) // 5
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
            width = x_max - x_min
            height = y_max - y_min
            assert width > 0
            assert height > 0
            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                result.append([width, height])
            # get k-means anchors on the original image size
            else:
                result.append([width, height])
    result = np.asarray(result)
    return result


def load_dataset(img_path, xml_path, target_size):
    xml_list = sorted(os.listdir(xml_path))
    dataset = []
    for index, xmlname in enumerate(xml_list):
        if xmlname.endswith('.xml'):
            xml = os.path.join(xml_path, xmlname)
            print('xml:', xml)
            sheet_dict = read_xml_to_json(xml)
            regions = sheet_dict['regions']
            img_size = sheet_dict['img_size']
            img_height = img_size[1]
            img_width = img_size[0]
            print('img_height:', img_height)
            print('img_width:', img_width)

            for bbox in regions:
                # img = cv2.imread(os.path.join(img_path, xmlname.replace('.xml', '.jpg')))
                # img_height, img_width = img.shape[:2]
                #
                # ratio_h = target_size[1] / img_height    # height
                # ratio_w = target_size[0] / img_width     # width

                # img_raw, img_transform, img_path1 = img_path_all[index]
                # img_height, img_width = img_raw.size
                ratio_h = target_size / img_height    # height
                ratio_w = target_size / img_width     # width

                width = int((bbox['bounding_box']['xmax'] - bbox['bounding_box']['xmin']) * ratio_w)
                height = int((bbox['bounding_box']['ymax'] - bbox['bounding_box']['ymin']) * ratio_h)
                # if width == 0 or height == 0:
                #     shutil.move(xml, r'E:\data\anchor\data\VOC2007\save\\' + xmlname)
                #     shutil.move(os.path.join(img_path, xmlname.replace('.xml', '.jpg')), r'E:\data\anchor\data\VOC2007\save\\' + xmlname.replace('.xml', '.jpg'))
                #     break

                dataset.append([width, height])
    print('dataset:', dataset)
    return np.array(dataset)


def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


def modify_xml_width_height_raw(xml_path, img_path, targets):
    datasets = DataSet(img_path, targets)
    xml_list = glob.glob(xml_path + os.sep + '*.xml')
    for index, xml in enumerate(xml_list):
        tree = ET.parse(xml)
        root = tree.getroot()
        img_raw, img_transform, img_path1 = datasets[index]
        try:
            if xml.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg').replace('\\', '/') == img_path1:
                # width = int(root.findall('size')[0].find('width').text)
                # height = int(root.findall('size')[0].find('height').text)
                img_width, img_height = img_raw.size
                size = root.findall('size')
                for obj in size:
                    width = obj.find('width')
                    width.text = str(img_width)

                    height = obj.find('height')
                    height.text = str(img_height)
                tree.write(xml)
                print(img_height)
        except Exception as e:
            print('xml:', xml)
            print('img_path1:', img_path1)


def modify_xml_width_height(xml_path, img_path, targets):
    datasets = DataSet(img_path, targets)
    xml_list = glob.glob(xml_path + os.sep + '*.xml')
    for index, xml in enumerate(xml_list):
        tree = ET.parse(xml)
        root = tree.getroot()
        img_raw, img_transform, img_path1 = datasets[index]
        try:
            # if xml.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg').replace('\\', '/') == img_path1:
                # width = int(root.findall('size')[0].find('width').text)
                # height = int(root.findall('size')[0].find('height').text)
            xml_name = xml.split('\\')[-1].replace('.xml', '')
            image_name = img_path1.split('/')[-1].replace('.jpg', '')
            if xml_name == image_name:
                img_width, img_height = img_raw.size
                size = root.findall('size')
                for obj in size:
                    width = obj.find('width')
                    width.text = str(img_width)

                    height = obj.find('height')
                    height.text = str(img_height)
                tree.write(xml)
                print(img_height)
        except Exception as e:
            print('xml:', xml)
            print('img_path1:', img_path1)
            continue


if __name__ == '__main__':
    target_size = 1600

    xml_path = r'E:\December\math_12_18\1_18\blank\en'
    img_path = r'E:\December\math_12_18\1_18\blank\en'

    # modify_xml_width_height(xml_path, img_path, target_size)


    # target resize format: [width, height]
    # if target_resize is speficied, the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale
    
    
    # target_size = 1120
    # # xml_path = 'data/all_math_blank/VOC2007/Annotations'
    # # img_path = 'data/all_math_blank/VOC2007/JPEGImages'

    anno_result = load_dataset(img_path, xml_path, target_size)
    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)
