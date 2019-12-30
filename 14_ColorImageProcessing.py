# -*- coding: utf-8 -*-
# @Time    : 2019/12/10 16:54
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import numpy as np
import cv2
img = cv2.imread('./picture/beach.png')

"""0. 彩色图像处理基础
        
        0. 功能：符合人眼视觉，简化目标物区分，根据颜色目标识别。
        1. 领域：全彩色，伪彩色。
        2. 描述光源质量：辐射率（放出能量），光强（接受能量），亮度（主管描述彩色强度，不好度量）
        3. 人眼敏感度：65%红，33%绿。2%蓝。
        4. 混色处理法：加色法（光，RGB相加为白色），减色法（颜料，RGB相加为黑色）
        5. 颜色特性：色调（Hue），饱和度（Saturation），亮度（Value）
"""

"""1. 彩色模型

        0. RGB：像素深度是每个像素的比特数，RGB一般是3*8=24Bytes深度，共可以表述2**24=16777216种颜色。
        1. CMY：青、深红、黄，用相减法。一般用于彩色打印机。
        2. HSV：V与彩色信息无关（将亮度与彩色信息分开，便于图像处理），HS与人眼感受相关。
        3. YCbCr：Y指亮度，Cb和Cr由(R-Y)和（B-Y）调整得到，用于JPEG等。
        4. 彩色转灰度，有两种方式，一个是直接（R+G+B）/3,一种是加权平均Y = 0.299R+0.587G+0.114B.
"""
# 0. RGB->HSV ; HSV->RGB
#    数学变换公式见该目录下RGB2HSV.png和HSV2RGB.png
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
# plt.subplot(121), plt.imshow(img_hsv)
# plt.subplot(122), plt.imshow(img_bgr)
# plt.show()


"""2. 彩色图像分割

      HSV空间：H描述擦色，S分离感兴趣特征区域。V一般不用。
      RGB空间：定义一个要分割的颜色（R0,G0,B0），如果颜色（Ri,Gi,Bi）与之欧氏距离很近，则认为近似。
"""
# 0. K均值在RGB空间的应用。
#    步骤：0. 任意选K个聚类中心，比如beach这个照片有蓝天海洋、树木、沙滩三个类别，那么K=3。
#          1. 按照最小距离分配像素点归属于哪一个聚类中心。
#          2. 计算各聚类中心新的向量值（该类型下所有像素的RGB均值）。
#          3. 计算代价函数J=所有像素到其聚类中心距离之和（这里距离指RGB or HSV距离）
#          4. 一直迭代到J收敛。
#    应用：以下是在RGB空间的应用（可以看出他把沙滩和海放在一个类别了，说明RGB分割效果不是很好，有时间再搞一个HSV分割）

# 0. 先随机生成K个中心
K = 3
centroid_r = np.random.randint(0, 255, [K, 1])
centroid_g = np.random.randint(0, 255, [K, 1])
centroid_b = np.random.randint(0, 255, [K, 1])
centroid_0 = np.concatenate((centroid_b, centroid_g, centroid_r), axis=1)[0]
centroid_1 = np.concatenate((centroid_b, centroid_g, centroid_r), axis=1)[1]
centroid_2 = np.concatenate((centroid_b, centroid_g, centroid_r), axis=1)[2]

# 1. 把K个中心复制img.shape[0]*img.shape[1]遍，也就是展开，为了之后矩阵操作更快
centroid_reshape_0 = np.tile(centroid_0, (img.shape[0] * img.shape[1], 1))
centroid_reshape_1 = np.tile(centroid_1, (img.shape[0] * img.shape[1], 1))
centroid_reshape_2 = np.tile(centroid_2, (img.shape[0] * img.shape[1], 1))

# 2. 把原图像也展开，而不是二维排列，为了和上面的质心矩阵一样size，矩阵操作更快
img_reshape = img.reshape(img.shape[0] * img.shape[1], 3)

# 3. dist用来存储每个像素RGB三个值和质心RGB的欧氏距离
dist = np.zeros([img.shape[0] * img.shape[1], K], dtype=np.float32)

# 4. 设定好初始的J(确保在第一轮迭代后，J_last=9999不会小于真正计算出来的J)，开始迭代
J_last = 99999
J = 9999
t = 10  # 运行次数，一般10次以内，不然直接退出
while J < J_last and t > 0:
    print(t)
    t = t-1
    J_last = J
    # 0.分别计算三个到质心的距离
    dist[:, 0] = np.linalg.norm(centroid_reshape_0 - img_reshape, axis=1)
    dist[:, 1] = np.linalg.norm(centroid_reshape_1 - img_reshape, axis=1)
    dist[:, 2] = np.linalg.norm(centroid_reshape_2 - img_reshape, axis=1)
    # 1. 取最小距离的那个质心作为该像素的标号
    label = np.argmin(dist, axis=1)
    index0 = np.argwhere(label == 0)
    index1 = np.argwhere(label == 1)
    index2 = np.argwhere(label == 2)
    # 2. 看看该质心下跟了多少个像素
    length0 = len(index0)
    length1 = len(index1)
    length2 = len(index2)
    # 3. 根据这些像素重新分配质心
    centroid_0 = np.round(1/length0*np.sum(img_reshape[np.squeeze(index0, axis=1)], axis=0))
    centroid_1 = np.round(1/length0*np.sum(img_reshape[np.squeeze(index1, axis=1)], axis=0))
    centroid_2 = np.round(1/length0*np.sum(img_reshape[np.squeeze(index2, axis=1)], axis=0))
    # 4. 把质心展开，方便矩阵操作
    centroid_reshape_0 = np.tile(centroid_0, (img.shape[0] * img.shape[1], 1))
    centroid_reshape_1 = np.tile(centroid_1, (img.shape[0] * img.shape[1], 1))
    centroid_reshape_2 = np.tile(centroid_2, (img.shape[0] * img.shape[1], 1))
    # 计算代价函数，在我们这里是所有像素到其质心的距离之和
    J0 = np.sum(np.linalg.norm(img_reshape[np.squeeze(index0, axis=1)] - np.tile(centroid_0, (length0, 1)), axis=1))
    J1 = np.sum(np.linalg.norm(img_reshape[np.squeeze(index1, axis=1)] - np.tile(centroid_1, (length1, 1)), axis=1))
    J2 = np.sum(np.linalg.norm(img_reshape[np.squeeze(index2, axis=1)] - np.tile(centroid_2, (length2, 1)), axis=1))
    J = (J0+J1+J2)/(img.shape[0]*img.shape[1])
# 5. 计算掩膜，准备画图
mask0 = np.zeros(img.shape, dtype=np.uint8)
mask1 = np.zeros(img.shape, dtype=np.uint8)
mask2 = np.zeros(img.shape, dtype=np.uint8)
for i in range(length0):
    mask0[np.squeeze(index0)[i]//img.shape[1], np.squeeze(index0)[i]%img.shape[1]] = 1
for i in range(length1):
    mask1[np.squeeze(index1)[i]//img.shape[1], np.squeeze(index1)[i]%img.shape[1]] = 1
for i in range(length2):
    mask2[np.squeeze(index2)[i]//img.shape[1], np.squeeze(index2)[i]%img.shape[1]] = 1
out0 = mask0*img
out1 = mask1*img
out2 = mask2*img
cv2.imshow('0', img)
cv2.imshow('1', out0)
cv2.imshow('2', out1)
cv2.imshow('3', out2)
cv2.waitKey(0)
cv2.destroyAllWindows()

