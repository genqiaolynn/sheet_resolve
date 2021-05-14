from PIL import Image, ImageEnhance
import os
import shutil
import numpy as np

'''
image enhance
'''


def image_enhance(img_path, xml_path, save_path):
    img_list = os.listdir(img_path)
    for img0 in img_list:
        if img0.endswith('.jpg'):
            img_name = img0.replace('.jpg', '')
            img_path0 = os.path.join(img_path, img0)
            img = Image.open(img_path0)
            img = img.convert("RGB")
            img = Image.fromarray(np.uint8(img))
            xml_path0 = os.path.join(xml_path, img0.replace('.jpg', '.xml'))
            shutil.copyfile(img_path0, os.path.join(save_path, img0))
            shutil.copyfile(xml_path0, os.path.join(save_path, img0.replace('.jpg', '.xml')))

            # color
            color_ratio = [0.5]
            for index, ratio in enumerate(color_ratio):
                im_1 = ImageEnhance.Color(img).enhance(ratio)
                save_name = img_name + '_' + str(index) + '_' + 'color' + '.jpg'
                img_save_path = os.path.join(save_path, save_name)
                im_1.save(img_save_path)
                xml_save_path = os.path.join(save_path, save_name.replace('.jpg', '.xml'))
                shutil.copyfile(os.path.join(xml_path, img0.replace('.jpg', '.xml')), xml_save_path)
            # Brightness
            # bright_ratio = [0.5]
            # for index, ratio in enumerate(bright_ratio):
            #     im_1 = ImageEnhance.Brightness(img).enhance(ratio)
            #     save_name = img_name + '_' + str(index) + '_' + 'bright' + '.jpg'
            #     img_save_path = os.path.join(save_path, save_name)
            #     im_1.save(img_save_path)
            #     xml_save_path = os.path.join(save_path, save_name.replace('.jpg', '.xml'))
            #     shutil.copyfile(os.path.join(xml_path, img0.replace('.jpg', '.xml')), xml_save_path)
            # Contrast
            # contrast_ratio = [0.8]
            # for index, ratio in enumerate(contrast_ratio):
            #     im_1 = ImageEnhance.Contrast(img).enhance(ratio)
            #     save_name = img_name + '_' + str(index) + '_' + 'contrast' + '.jpg'
            #     img_save_path = os.path.join(save_path, save_name)
            #     im_1.save(img_save_path)
            #     xml_save_path = os.path.join(save_path, save_name.replace('.jpg', '.xml'))
            #     shutil.copyfile(os.path.join(xml_path, img0.replace('.jpg', '.xml')), xml_save_path)
            # Sharpness
            sharp_ratio = [0.5]
            for index, ratio in enumerate(sharp_ratio):
                im_1 = ImageEnhance.Sharpness(img).enhance(ratio)
                save_name = img_name + '_' + str(index) + '_' + 'sharp' + '.jpg'
                img_save_path = os.path.join(save_path, save_name)
                im_1.save(img_save_path)
                xml_save_path = os.path.join(save_path, save_name.replace('.jpg', '.xml'))
                shutil.copyfile(os.path.join(xml_path, img0.replace('.jpg', '.xml')), xml_save_path)
                print(xml_save_path)
    print('ok')


if __name__ == '__main__':
    img_path = r'E:\December\math_12_18\1_18\unblank\raw'
    save_path = r'E:\December\math_12_18\1_18\unblank\en'
    # xml_path = img_path
    xml_path = r'E:\December\math_12_18\1_18\unblank\raw'
    image_enhance(img_path, xml_path, save_path)

