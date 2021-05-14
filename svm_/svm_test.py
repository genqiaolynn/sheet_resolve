# coding:UTF-8

import numpy as np
# import _pickle as pickle
import pickle
from svm_.svm import svm_predict
from svm_.svm import SVM
from svm_.SVM_Train import get_feature


def load_test_data(test_file):
    data = []
    f = open(test_file)
    for line in f.readlines():
        lines = line.strip().split(' ')

        # 处理测试样本中的特征
        index = 0
        tmp = []
        for i in range(0, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while int(li[0]) - 1 > index:
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data)


def load_svm_model(svm_model_file):
    with open(svm_model_file, 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model


def get_prediction(test_data, svm):
    m = np.shape(test_data)[0]
    prediction = []
    for i in range(m):
        # 对每一个样本得到预测值
        predict = svm_predict(svm, test_data[i, :])
        # 得到最终的预测类别
        prediction.append(str(np.sign(predict)[0, 0]))
    print('prediction:', prediction)
    return prediction


def get_prediction_single_img(test_data, svm):
    predict = svm_predict(svm, test_data)
    # 得到最终的预测类别
    prediction = str(np.sign(predict)[0, 0])
    label_dict = {'1.0': 'blank', '-1.0': 'unblank'}
    print('prediction:', label_dict[prediction])
    return prediction


def save_prediction(result_file, prediction):
    f = open(result_file, 'w')
    f.write(" ".join(prediction))
    f.close()


if __name__ == "__main__":
    # 1、导入测试数据
    print("--------- 1.load data ---------")
    # test_data = load_test_data("svm_test_data")
    img_path = r'E:\December\math_12_18\1_18\mix1\2.jpg'
    feature = get_feature(img_path)
    # print(test_data.shape)
    # 2.Softmax Regression Train、导入SVM模型
    print("--------- 2.Softmax Regression Train.load model ----------")
    svm_model = load_svm_model(r"F:\SVM_SMO\model_file.npy")
    # 3、得到预测值
    print("--------- 3.get prediction ---------")
    prediction = get_prediction_single_img(np.asarray(feature), svm_model)
    # prediction = get_prediction(test_data, svm_model)
    # 4、保存最终的预测值
    print("--------- 4.save result ----------")
    save_prediction("result", prediction)