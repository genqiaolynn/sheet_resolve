# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/20 8:44'


import numpy as np
import re, cv2
from brain_api_charge import get_ocr_text_and_coordinate0
import xml.etree.ElementTree as ET
from utils import create_xml


'''
单张答题卡ocr，根据分栏点将ocr的内容分栏，栏内y轴排序
总结这个文件:
1. 先把words的坐标转换成[left, top, right, bottom, xmid, ymid]的格式
2. xmid排序
3. 拿到所有的xmid，并根据分栏的结果将ocr的内容对应分到每栏内
4. 拿到所有中文行的索引值
5. 找栏内跳跃的中文索引值，两个索引值之间的内容就是一个block
6. 对block进行分上界下界的时候需要注意一点，我拿到的是中文索引值，中文的上面或者下面才是区域的地方，
所以要考虑上界的下一行，下界的上一行，这两者之间的才是想要的区域
'''


# ocr2sheet1是使用的words的坐标作为标准去分栏，当ocr识别结果连横跨栏的时候，文字分栏会产生错误，考虑用单个字符的坐标去改进
# 单个字符改进是OK的，能解决问题，单独写在下面的函数中
def ocr2sheet(image, col_split_list, raw_ocr):
    image_y, image_x = image.shape[0], image.shape[1]
    for words_string in raw_ocr:
        words = words_string['words']
        loc = words_string['location']
        right = loc['left'] + loc['width']
        bottom = loc['top'] + loc['height']
        xmid = loc['left'] + loc['width'] // 2
        ymid = loc['top'] + loc['height'] // 2
        loc.update({'right': right, 'bottom': bottom, 'xmid': xmid, 'ymid': ymid})
    print(raw_ocr)
    raw_ocr_len = len(raw_ocr)
    raw_ocr = sorted(raw_ocr, key=lambda k: k['location']['xmid'])
    xmid_list = [ele['location']['xmid'] for ele in raw_ocr]

    col_list = []
    for split in col_split_list:
        xmid_list.append(split)
        xmid_list = sorted(xmid_list)
        split_index = xmid_list.index(split)
        col_list.append(raw_ocr[:split_index])
        raw_ocr = raw_ocr[split_index:]
        xmid_list = xmid_list[split_index+1:]
    # 这时候的raw_ocr是指切片剩余的部分，已经不再 是原始的ocr list了
    if raw_ocr:
        col_list.append(raw_ocr)
    # 截止到这部分，已经将整张答题卡ocr的内容放在每个块内了

    col_split_list.insert(0, 1)
    col_split_list.append(image_x)

    # 分析块内，找中文区域
    zh_char = r'[\u4E00-\u9FA5]'   # 单个字符的中文    要是想匹配多个字符的中文后面加上+
    zh_char_list = [zh_char]
    punctuation_p = '[，；：。,;:·√()（）]+'
    block_list = []
    for ii, ocr_res in enumerate(col_list):
        ocr_res = sorted(ocr_res, key=lambda k: k['location']['top'])
        raw_chn_index = []
        for i, words_line in enumerate(ocr_res):
            words = words_line['words']
            words_num_list = []
            width = words_line['location']['width']
            height = words_line['location']['height']
            if width >= height:    # 一句中文的宽要大于字高
                for p in zh_char_list:
                    words_m = re.finditer(p, words)
                    words_list = [(ele.group(), ele.span()) for ele in words_m if ele]
                    words_num = len(words_list) * 2
                    words_num_list.append(words_num)
                if sum(words_num_list) >= 2:
                    raw_chn_index.append(i)
        print(raw_chn_index)

        chn_index = raw_chn_index.copy()
        left_limit, right_limit = col_split_list[ii], col_split_list[ii+1]
        if ocr_res:
            left_limit = min([ele['location']['left'] for ele in ocr_res]) - 10
            right_limit = max([ele['location']['right'] for ele in ocr_res]) + 10
        # 这边取的是words的坐标，但是ocr过程中会出现连横跨栏的情况，使的分栏的结果不准确
        left_limit = max(left_limit, col_split_list[ii])
        right_limit = min(right_limit, col_split_list[ii + 1])

        # 找出栏内分割点跳跃的点
        if len(ocr_res) - 1 not in chn_index:
            chn_index.append(len(ocr_res)-1)
        chn_index_arr = np.array(chn_index)
        numbers_interval = abs(chn_index_arr[1:] - chn_index_arr[:-1])
        # 始终存的是索引值,存的是索引值出异常的点的索引
        split_index_tmp = []
        for i, interval in enumerate(numbers_interval):
            if interval > 2 and interval > np.mean(np.array(numbers_interval)):
                split_index_tmp.append(i)

        split_index_tmp = sorted(list(set(split_index_tmp)))
        print(split_index_tmp)

        # 确定每个block的上界和下界
        for i, ele in enumerate(split_index_tmp):
            top_limit = chn_index[ele]
            if top_limit == len(ocr_res) - 1:   # 如果上界等于下界则退出
                break
            else:
                # 找上界的下一个索引对应的索引为下界，两个中文行之间为一个block
                bottom_limit = chn_index[split_index_tmp[i] + 1]
                if bottom_limit in chn_index:
                    # 这一行要去掉是字符的情况，也要去掉是标点符号的情况
                    while(ocr_res[bottom_limit - 1]['location']['height'] >= ocr_res[bottom_limit - 1]['location']['width']
                    or ocr_res[bottom_limit - 1]['words'] in punctuation_p):
                        bottom_limit = bottom_limit - 1
                    bottom = ocr_res[bottom_limit - 1]['location']['top'] + ocr_res[bottom_limit - 1]['location']['height']
                else:
                    # 这个情况只能是最后一个索引，其他的索引都在chn_index这个list里
                    bottom_limit = chn_index[-1]
                    bottom = ocr_res[bottom_limit]['location']['top'] + ocr_res[bottom_limit]['location']['height']
                top = ocr_res[top_limit + 1]['location']['top']   # 上界的下一行肯定是没问题的，上界在的那一行会把字包进来
                left = left_limit
                right = right_limit
                block_list.append({'loc': [left, top, right, bottom]})

    # xml_template = 'exam_info/000000-template.xml'
    # tree = ET.parse(xml_template)
    # for i, ele in enumerate(block_list):
    #     create_xml(f'block{i}', tree, ele['loc'][0], ele['loc'][1], ele['loc'][2], ele['loc'][3])
    # tree.write(r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\2.xml')
    return block_list


def ocr2sheet_single_char(image, col_split_list, raw_ocr, xml_path=None):
    image_y, image_x = image.shape[0], image.shape[1]
    single_char_ocr = []
    for ele in raw_ocr:
        for words_string in ele['chars']:
            loc = words_string['location']
            right = loc['width'] + loc['left']
            bottom = loc['height'] + loc['top']
            xmid = loc['left'] + loc['width'] // 2
            ymid = loc['top'] + loc['height'] // 2
            loc.update({'right': right, 'bottom': bottom, 'xmid': xmid, 'ymid': ymid})
            single_char_ocr.append(words_string)
    print(single_char_ocr)
    # 选用单个字符能避免文字识别结果连横跨栏的情况，可以使用ocr进行分栏
    single_char_ocr = sorted(single_char_ocr, key=lambda k: k['location']['xmid'])
    x_mid_list = sorted([ele['location']['xmid'] for ele in single_char_ocr])
    col_list = []
    for split in col_split_list:
        x_mid_list.append(split)
        x_mid_list = sorted(x_mid_list)
        split_index = x_mid_list.index(split)
        col_list.append(single_char_ocr[:split_index])
        single_char_ocr = single_char_ocr[split_index:]           # 这边split_index不加1的原因是分栏点本身也没有东西，一般分栏点是不带文字信息的，所以无需跳过去
        x_mid_list = x_mid_list[split_index+1:]   # 这边split_index+1的原因是split_index本身是分栏点，需要跳过这个往下找
    if single_char_ocr:
        col_list.append(single_char_ocr)
    col_split_list.insert(0, 1)
    col_split_list.append(image_x)

    block_list = []
    # eng_char_p = r'[u4e00-u9fa5]+'     # english
    chn_char_p = r'[\u4E00-\u9FA5]'      # 单个字符的匹配   # chinese
    chn_char_p_m = r'[\u4E00-\u9FA5]+'   # 多字符的匹配     # chinese
    chn_char_p_list = [chn_char_p]

    xml_template = 'exam_info/000000-template.xml'
    tree = ET.parse(xml_template)
    for ii, ocr_res in enumerate(col_list):
        ocr_res = sorted(ocr_res, key=lambda k: k['location']['top'])
        raw_chh_index = []
        for i, eee in enumerate(ocr_res):
            loc = eee['location']
            width = loc['width']
            height = loc['height']
            create_xml(eee['char'], tree, eee['location']['left'], eee['location']['top'],
                           eee['location']['left'] + eee['location']['width'], eee['location']['top'] + eee['location']['height'])
        tree.write(r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\2.xml')
        print('ok')


if __name__ == '__main__':
    img_path = r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-02-20\1.jpg'
    img = cv2.imread(img_path)
    col_split_list = [1768, 3281]
    raw_ocr = get_ocr_text_and_coordinate0(img)
    ocr2sheet(img, col_split_list, raw_ocr)