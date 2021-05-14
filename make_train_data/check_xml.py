# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/3/6 11:40'


from utils import *
import shutil
import cv2
from PIL import Image

# txt_path = r'F:\yolov3_\anchor\data\labels'
# save_path = r'F:\yolov3_\anchor\data\save'
# txt_list = os.listdir(txt_path)
# for ele in txt_list:
#     txt_path0 = os.path.join(txt_path, ele)
#     ff = open(txt_path0, 'r', encoding='utf-8').read()
#     if len(ff) <= 3:
#         shutil.copyfile(txt_path0, os.path.join(save_path, ele))
#     else:
#         continue
#     print(ff)


# xml_path = r'E:\data\anchor\data\VOC2007\Annotations'
# xml_list = os.listdir(xml_path)
# xx = []
# for ele in xml_list:
#     xml = os.path.join(xml_path, ele)
#     tree = ET.parse(xml)
#     sheet_dict = read_xml_to_json(xml)
#     regions = sheet_dict['regions']
#     xx.append(len(regions))
#     # if len(regions) == 1:
#     #     shutil.move(xml, r'F:\yolov3_\anchor\data\VOC2007\save\\' + ele)
#     #     shutil.move(r'F:\yolov3_\anchor\data\VOC2007\JPEGImages\\' + ele.replace('.xml', '.jpg'),
#     #                 r'F:\yolov3_\anchor\data\VOC2007\save\\' + ele.replace('.xml', '.jpg'))
#     # print(regions)
# print(xx)

# def convert_pil_to_jpeg(raw_img):
#     if raw_img.mode == 'L':   # L是二值图像
#         channels = raw_img.split()
#         img = Image.merge("RGB", (channels[0], channels[0], channels[0]))
#     elif raw_img.mode == 'RGB':
#         img = raw_img
#     elif raw_img.mode == 'RGBA':
#         img = Image.new("RGB", raw_img.size, (255, 255, 255))
#         img.paste(raw_img, mask=raw_img.split()[3])  # 3 is the alpha channel
#     elif raw_img.mode == 'P':
#         img = raw_img.convert('RGB')
#     else:
#         img = raw_img
#     open_cv_image = np.array(img)
#     return img, open_cv_image
#
#
# img_path = r'E:\data\anchor\data\anchor_dataset\images'
# img_list = os.listdir(img_path)
# for ele in img_list:
#     img_path0 = os.path.join(img_path, ele)
#     image = Image.open(img_path0)
#     image = convert_pil_to_jpeg(image)[0]
#     image.save(r'E:\data\anchor\data\anchor_dataset\save_img\\' + ele)



# xml_path = r'E:\data\anchor\data\VOC2007\Annotations'
# xml_list = os.listdir(xml_path)
# xx = []
# for ele in xml_list:
#     xml = os.path.join(xml_path, ele)
#     tree = ET.parse(xml)
#     sheet_dict = read_xml_to_json(xml)
#     regions = sheet_dict['regions']
#     xx.append(len(regions))
#     if len(regions) == 58:
#         shutil.copyfile(xml, r'E:\data\anchor\data\VOC2007\save\\' + ele)
#         shutil.copyfile(r'E:\data\anchor\data\VOC2007\JPEGImages\\' + ele.replace('.xml', '.jpg'),
#                     r'E:\data\anchor\data\VOC2007\save\\' + ele.replace('.xml', '.jpg'))
#     print(regions)
# print(xx)


txt_path = r'E:\data\anchor\data\anchor_dataset\labels'
txt_list = os.listdir(txt_path)
for ele in txt_list:
    txt_path0 = os.path.join(txt_path, ele)
    ff = np.loadtxt(txt_path0)
    print(ff)