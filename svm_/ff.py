# @Author  : lightXu
# @File    : predict.py
import os
import time

import cv2
import numpy as np

model = np.load('model.npy')   # model是个(10, 21)的矩阵


def round_float(value):
    return round(value, 6)


def gen_feature(img):

    h, w = img.shape
    mean = round_float(np.mean(img)/255)    # 四舍五入取六位   图像每个像素归一化后取均值
    aspect = round_float(h / w)   # 取外观的比例
    # mean和aspect两个指标这么做的目的是使的指标处于同一个量级，方便后续的计算

    row_box, col_box = 4, 4
    row_box_pix, col_box_pix = h // 4, w // 4

    box_feature = [aspect, mean]
    for r in range(row_box):
        for c in range(col_box):
            box = img[r*row_box_pix:(r+1)*row_box_pix, c*col_box_pix:(c+1)*col_box_pix]
            box_mean = round_float(np.mean(box)/255)
            box_feature.append(box_mean)

    return box_feature


def kernel_trans(x1, array, k_tup=('rbf', 1.3)):
    if k_tup[0] == 'lin':
        kernel = np.dot(x1, array)
    elif k_tup[0] == 'rbf':
        sigma = k_tup[1]
        if np.ndim(x1) == 1:
            kernel = np.exp(np.sum(np.square(x1 - array)) / (-1 * np.square(sigma)))
        else:
            kernel = np.sum(np.square(x1-array), axis=1)
            kernel = np.exp(-kernel / np.square(sigma))
    else:
        raise NameError('核函数无法识别')

    return kernel


def svm_predict(img, subject_id=3):
    feature = gen_feature(img)
    feature.insert(0, subject_id)   # insert的作用是在索引值的位置插入一个数   insert(index, ele)  index是指索引值   ele数

    feature = np.array(feature, dtype=np.float)

    # model = np.load('model.npy')
    alpha_y = model[:, 0].T
    b = model[0, 1]
    sVs = model[:, 2:]

    fxk = np.dot(alpha_y, kernel_trans(sVs, feature)) + b
    p = np.sign(fxk)
    if p == 1:
        print('blank')
        # return True
    else:
        print('unblank')
        # return False


if __name__ == '__main__':
    file = r'F:\exam\exam_segment_django_9_18\segment\exam_image\sheet\math\2021-01-29\1.jpg'
    im = cv2.imread(file, 0)
    t1 = time.time()
    svm_predict(im)
    t2 = time.time()
    print(t2-t1)
