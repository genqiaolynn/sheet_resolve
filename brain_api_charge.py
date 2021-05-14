# @Author  : lightXu
# @File    : brain_api.py
# @Time    : 2018/11/21 0021 下午 16:20
import requests
import base64
from urllib import parse, request
import cv2, json
import glob
import numpy as np
from PIL import Image


OCR_BOX_URL = 'https://aip.baidubce.com/rest/2.0/ocr/v1/'
OCR_URL = 'https://aip.baidubce.com/rest/2.0/ocr/v1/'
# OCR_ACCURACY = 'general'
OCR_ACCURACY = 'accurate'
OCR_CLIEND_ID = 'AVH7VGKG8QxoSotp6wG9LyZq'
OCR_CLIENT_SERCERT = 'gG7VYvBWLU8Rusnin8cS8Ta4dOckGFl6'
OCR_TOKEN_UPDATE_DATE = 10


def get_access_token(OCR_CLIEND_ID, OCR_CLIENT_SERCERT):
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s' \
           % (OCR_CLIEND_ID, OCR_CLIENT_SERCERT)
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    req = request.Request(method='GET', url=host, headers=headers)
    response = request.urlopen(req)
    if (response.status == 200):
        data = json.loads(response.read().decode())
        access_token = data['access_token']
        return access_token


def opecv2base64(img):
    image = cv2.imencode('.jpg', img)[1]
    base64_data = str(base64.b64encode(image))[2:-1]
    return base64_data


def get_ocr_text_and_coordinate0(img, ocr_accuracy=OCR_ACCURACY, language_type='CHN_ENG'):
    textmod = {'access_token': get_access_token(OCR_CLIEND_ID, OCR_CLIENT_SERCERT)}
    textmod = parse.urlencode(textmod)
    url = '{}{}{}{}'.format(OCR_BOX_URL, ocr_accuracy, '?', textmod)
    url_general = '{}{}{}{}'.format(OCR_BOX_URL, 'general', '?', textmod)

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    image_type = 'base64'
    group_id = 'group001'
    user_id = 'usr001'

    image = opecv2base64(img)

    data = {
        'image_type': image_type,
        'group_id': group_id,
        'user_id': user_id,
        'image': image,
        'detect_direction': 'true',
        'recognize_granularity': 'small',
        'language_type': language_type,
        # 'vertexes_location': 'true',
        # 'probability': 'true'
    }

    # resp = requests.post(url, data=data, headers=headers, timeout=15).json()
    resp = requests.post(url, data=data, headers=headers).json()
    if resp.get('error_msg'):
        if 'internal error' in resp.get('error_msg'):
            resp = requests.post(url_general, data=data, headers=headers).json()
            if resp.get('error_msg'):
                raise Exception("ocr {}!".format(resp.get('error_msg')))
        else:
            raise Exception("ocr {}!".format(resp.get('error_msg')))

    words_result = resp.get('words_result')
    # print(words_result)

    # string1 = ''
    # for ele in words_result:
    #     for ele1 in ele['chars']:
    #         char = ele1['char']
    #         string1 = string1 + ',' + char
    # print(string1)
    # print(resp)
    # for ele in words_result:
    #     words = ele['words']
    #     print(words)
    return words_result


def get_ocr_text_and_coordinate(img):
    textmod = {'access_token': get_access_token(OCR_CLIEND_ID, OCR_CLIENT_SERCERT)}
    textmod = parse.urlencode(textmod)
    url = '{}{}{}{}'.format(OCR_BOX_URL, OCR_ACCURACY, '?', textmod)
    url_general = '{}{}{}{}'.format(OCR_BOX_URL, 'general', '?', textmod)

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    image_type = 'base64'
    group_id = 'group001'
    user_id = 'usr001'

    image = opecv2base64(img)

    data = {
        'image_type': image_type,
        'group_id': group_id,
        'user_id': user_id,
        'image': image,
        'detect_direction': 'false',
        'recognize_granularity': 'small',
        # 'vertexes_location': 'true',
        # 'probability': 'true'
    }
    try:
        resp = requests.post(url, data=data, headers=headers).json()
        if resp.get('error_msg'):
            if 'internal error' in resp.get('error_msg'):
                resp = requests.post(url_general, data=data, headers=headers).json()
                if resp.get('error_msg'):
                    raise Exception("ocr {}!".format(resp.get('error_msg')))
            else:
                raise Exception("ocr {}!".format(resp.get('error_msg')))

        # words_result = resp.get('words_result')
        # save_path = img0.replace('.jpg', '.txt')
        # save_path = r'H:\12_30_data\result\result.txt'
        # with open(save_path, "a") as f:
        #     for line in resp["words_result"]:
        #         print(line["words"], end="")
        #         f.write(line["words"] + "\n")
        # f.close()
        # return words_result
        return resp
    except Exception as e:
        pass


def get_ocr_text_and_coordinate11(img, ocr_accuracy=OCR_ACCURACY, language_type='CHN_ENG'):
    textmod = {'access_token': get_access_token(OCR_CLIEND_ID, OCR_CLIENT_SERCERT)}
    textmod = parse.urlencode(textmod)
    url = '{}{}{}{}'.format(OCR_BOX_URL, ocr_accuracy, '?', textmod)
    url_general = '{}{}{}{}'.format(OCR_BOX_URL, 'general', '?', textmod)

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    image_type = 'base64'
    group_id = 'group001'
    user_id = 'usr001'

    image = opecv2base64(img)

    data = {
        'image_type': image_type,
        'group_id': group_id,
        'user_id': user_id,
        'image': image,
        'detect_direction': 'true',
        'recognize_granularity': 'small',
        'language_type': language_type,
        # 'vertexes_location': 'true',
        # 'probability': 'true'
    }

    # resp = requests.post(url, data=data, headers=headers, timeout=15).json()
    resp = requests.post(url, data=data, headers=headers).json()
    if resp.get('error_msg'):
        if 'internal error' in resp.get('error_msg'):
            resp = requests.post(url_general, data=data, headers=headers).json()
            if resp.get('error_msg'):
                raise Exception("ocr {}!".format(resp.get('error_msg')))
        else:
            raise Exception("ocr {}!".format(resp.get('error_msg')))

    words_result = resp['words_result']
    return words_result


def png2jpg(png_path):
    try:
        im = Image.open(png_path)
        jpg_path = png_path.replace('.png', '.jpg')
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, im)
        bg.save(jpg_path)
        return jpg_path
    except Exception as e:
        print("PNG转换JPG 错误", e)


if __name__ == "__main__":
    # img_path = r'H:\12_30_data\type_score'
    # img_list = glob.glob(img_path + '\\*.jpg')
    # for img0 in img_list:
    #     img = np.asarray(cv2.imread(img0))
    #     get_ocr_text_and_coordinate(img, img0)

    # img_path = r'C:\Users\Administrator\Desktop\11\11.png'
    # if img_path.endswith('.png'):
    #     img_path = png2jpg(img_path)
    #     img = np.asarray(cv2.imread(img_path))
    #     get_ocr_text_and_coordinate(img, img_path)


    # img_path = r'F:\exam_segment_django113\segment\exam_image\sheet\science_comprehensive\2020-01-13\choice_region.jpg'
    img_path = r'E:\7_2_test_img\2\11.jpg'
    img = np.asarray(cv2.imread(img_path))
    get_ocr_text_and_coordinate0(img)