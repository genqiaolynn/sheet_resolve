import os
import cv2
import imutils
from matplotlib import pyplot as plt
import numpy as np
#import kobe
import time
import datetime

def plt_imshow(image_gray):
    plt.imshow(image_gray,cmap='gray')
    plt.show()
def plt_imshow_bgr(image):
    plt.imshow(imutils.opencv2matplotlib(image))
    plt.show()
def plt_imshow_bgr_image(image):
    plt.imshow(image)
    plt.show()
#全局阈值
def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
    print("threshold value %s"%ret)
    return binary

#局部阈值
def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #把输入图像灰度化
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 25, 10)
    return binary

#用户自己计算阈值
def custom_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY_INV)
    return binary
# 基本全局阈值计算
def basic_global_threshold(hist):
    #i,t,t1,t2,k1,k2
    t=0;u=0;
    for idx,val in enumerate(hist):
        t+=val
        u+=idx*val
    k2 = int(u/t)
    k1= 0
    while(k1!=k2):
        k1=k2
        t1=0;u1=0;
        for idx,val in enumerate(hist):
            if(idx>k1):
                break
            t1+=val
            u1+=idx*val
        t2=t-t1
        u2=u-u1
        if t1: u1=u1/t1
        else: u1=0
        if t2: u2=u2/t2
        else: u2=0
        k2=int((u1+u2)/2)
    return k1
def custom_basic_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #把输入图像灰度化
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_arr = np.asarray(hist).astype(np.int32).flatten()
    mean = basic_global_threshold(hist_arr)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY_INV)
    return binary,mean

testPath = r"F:\data\test-data\201907141524_0002.jpg"
print(testPath)
img = cv2.imread(testPath)
imH, imW =img.shape[:2]
imgGray = cv2.imread(testPath,0)
imgBin,thresholdV = custom_basic_threshold(img)
ret,imgBinOne = cv2.threshold(imgGray, thresholdV, 1, cv2.THRESH_BINARY_INV)
imgIter = cv2.integral(imgBinOne) #积分图
print(imgIter)
cv2.imwrite(r"F:\data\test-data\bin.tif",imgBin)
cv2.imwrite(r"F:\data\test-data\jft.png",imgIter)
#plt.figure(figsize=(10,10))
#plt.subplot(211)
#plt.imshow(imutils.opencv2matplotlib(img))
#plt.subplot(212)
#plt.imshow(imgBin,cmap='gray')
#plt.show()

#crossPoints = [(364, 319) ,(3169,169),(3172,2122),(363,2116)]
crossPoints = [(817, 496), (869, 496), (923, 496), (976, 496), (1392, 496), (363, 527), (448, 527), (2607, 655), (1733, 670), (1420, 810), (2284, 810), (363, 823), (448, 823), (501, 823), (553, 823), (606, 823), (661, 823), (711, 823), (764, 823), (817, 823), (869, 823), (923, 823), (976, 823), (1392, 823), (363, 846), (468, 846), (614, 846), (1227, 846), (1392, 846), (1840, 865), (1629, 865), (1716, 865), (614, 902), (736, 902), (764, 496), (860, 902), (983, 902), (1104, 902), (1227, 902), (1629, 920), (1716, 920), (1840, 920), (363, 957), (468, 957), (614, 957), (736, 957), (860, 957), (983, 957), (1104, 957), (1227, 957), (1392, 957), (2672, 957), (1629, 976), (1716, 976), (1840, 976), (363, 1012), (468, 1012), (614, 1012), (736, 1012), (860, 1012), (983, 1012), (1104, 1012), (1227, 1012), (1392, 1012), (3167, 1025), (1556, 1062), (363, 1102), (1392, 1102), (2732, 1310), (1698, 1444), (363, 1501), (1392, 1501), (1578, 1522), (1926, 1522), (2712, 1540), (363, 1588), (1392, 1588), (576, 1609), (661, 1609), (785, 1609), (3060, 1616), (576, 1664), (661, 1664), (785, 1664), (576, 1719), (661, 1719), (785, 1719), (628, 1798), (2178, 1826), (661, 1877), (880, 1877), (1256, 1877), (1783, 1904), (801, 1952), (1629, 1980), (363, 2071), (1420, 2071), (2284, 2071), (2312, 2071), (3172, 2071), (363, 2120), (2284, 2120), (2312, 2120), (3172, 2120), (1420, 166), (2284, 166), (2312, 166), (3167, 166), (1420, 206), (3167, 206), (2740, 228), (2618, 228), (2530, 282), (2618, 282), (2740, 282), (363, 321), (448, 321), (976, 321), (2530, 338), (2618, 338), (2740, 338), (448, 400), (501, 400), (553, 400), (661, 400), (711, 400), (976, 400), (764, 400), (817, 400), (869, 400), (923, 400), (448, 496), (501, 496), (553, 496), (606, 496), (661, 496), (711, 496)]
clen = 20  # 交叉点边线长度
ch = 2  # 误差宽度 一半
adjust = 15  # 交叉点微调区域
preSplit = 0.8
totalPixes = clen
print(totalPixes)
crossBox = []
npImg = np.array(imgBinOne)
npJft = np.array(imgIter)
print(imW, imH)

class CrossPt:
    def __init__(self):
        self.point = ()
        self.way = []
        self.confidence = 0.0

CrossPt_list = []

start = datetime.datetime.now()
print(start)

for point in crossPoints:
    #print(point)
    left = (point[0] - adjust) if (point[0] - adjust) > 0 else 0
    top = (point[1] - adjust) if (point[1] - adjust) > 0 else 0
    right = (point[0] + adjust) if (point[0] + adjust) < imW else imW - 1
    bottom = (point[1] + adjust) if (point[1] + adjust) < imH else imH - 1
    crossBox.append((left, top, right, bottom))
for box in crossBox:
    #print(box)
    # 遍历
    # imgCrop = kobe.image_roi(imgGray,box)
    # plt_imshow(imgCrop)
    fcount = 0

    #imgOne = kobe.image_roi(imgBinOne, box)
    #plt_imshow(imgOne)
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            v = npImg[y][x]
            if (v == 0):
                continue
            # imgCor= cv2.circle(img,(x,y),1,(0,0,244),1,8,0)
            fcount += 1
            # 上下左右判断
            flag = [0, 0, 0, 0]
            trueCount = 0
            fri = [0.0, 0.0, 0.0, 0.0]

            # up
            left = x - ch  # 这里应该有安全判断 暂时不做
            top = y - clen
            right = x + ch
            bottom = y
            oneCount = 0
            for n in range(top, bottom+1):
                fontCount = npJft[n+1][right] + npJft[n][left] - npJft[n][right] - npJft[n+1][left]
                if (fontCount != 0):
                    oneCount += 1
            fri[0] = oneCount / totalPixes
            if (fri[0] > preSplit):
                flag[0] = 1
                trueCount += 1

            # down
            left = x - ch  # 这里应该有安全判断 暂时不做
            top = y
            right = x + ch
            bottom = y + clen
            oneCount = 0
            for n in range(top, bottom):
                fontCount = npJft[n+1][right] + npJft[n][left] - npJft[n][right] - npJft[n+1][left]
                if (fontCount != 0):
                    oneCount += 1
            fri[1] = oneCount / totalPixes
            if (fri[1] > preSplit):
                flag[1] = 1
                trueCount += 1

            # left
            left = x - clen  # 这里应该有安全判断 暂时不做
            top = y - ch
            right = x
            bottom = y + ch
            oneCount = 0
            for n in range(left, right+1):
                fontCount = npJft[bottom][n+1] + npJft[top][n] - npJft[top][n+1] - npJft[bottom][n]
                if (fontCount != 0):
                    oneCount += 1
            fri[2] = oneCount / totalPixes
            if (fri[2] > preSplit):
                flag[2] = 1
                trueCount += 1

            # right
            left = x  # 这里应该有安全判断 暂时不做
            top = y - ch
            right = x + clen
            bottom = y + ch
            oneCount = 0
            for n in range(left, right):
                fontCount = npJft[bottom][n+1] + npJft[top][n] - npJft[top][n+1] - npJft[bottom][n]
                if (fontCount != 0):
                    oneCount += 1
            fri[3] = oneCount / totalPixes
            if (fri[3] > preSplit):
                flag[3] = 1
                trueCount += 1

            if(trueCount < 2):
                continue
            elif(trueCount==2):
                if((flag[0] and flag[1]) or (flag[2] and flag[3])):
                    continue
            if (trueCount >= 2):
                #print(x,y ,fri," this is a cp")
                cp = CrossPt()
                cp.point = (x,y)
                cp.way = flag
                cp.confidence = fri[0]+fri[1]+fri[2]+fri[3]
                #print("xx" ,cp.point, cp.way, cp.confidence)
                CrossPt_list.append(cp)
                #imgCor= cv2.circle(img,(x,y),1,(0,0,244),1,8,0)
end = datetime.datetime.now()
print(start)
print(end)
print(fcount , "time : " ,end-start)

imgCor = img
print(len(CrossPt_list))

for cross in CrossPt_list:
    #print(cross.point,cross.way,cross.confidence)
    cv2.circle(imgCor,cross.point,clen,(0,0,233),2,8,0)
    if(cross.way[0]):
        cv2.line(imgCor,cross.point,(cross.point[0],cross.point[1]-clen),(0,0,244),2,8,0)
    if (cross.way[1]):
        cv2.line(imgCor, cross.point, (cross.point[0], cross.point[1] + clen), (0, 0, 244),2, 8, 0);
    if (cross.way[2]):
        cv2.line(imgCor, cross.point, (cross.point[0] - clen, cross.point[1] ), (0, 0, 244),2, 8, 0);
    if (cross.way[3]):
        cv2.line(imgCor, cross.point, (cross.point[0] + clen, cross.point[1] ), (0, 0, 244),2, 8, 0);

cv2.imwrite(r"F:\data\test-data\cor.png", imgCor)
plt_imshow(imgCor)



if __name__ == '__main__':
    print("this is test!")












