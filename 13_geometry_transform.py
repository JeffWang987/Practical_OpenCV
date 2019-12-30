# -*- coding: utf-8 -*-
# @Time    : 2019/12/8 13:17
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./picture/dinosaur.jpg', 0)
"""
0. 一些基础知识
    齐次坐标：用n＋1维向量表示n维向量的方法称为齐次坐标表示法。
    规范化齐次坐标：H＝1时，则(x, y, 1)就称为点(x, y)的规范化齐次坐标。
    变换矩阵T（针对齐次坐标）： [[a b p]   ，其中左上角abcd实现恒等、镜像、错且和旋转；
                                [c d q]        右上角pq实现平移变换；
                                [l m s]]       左下角lm实现透视变换；右下角s实现图像的全比例变换（规范齐次坐标）。
"""

"""
1. 比例缩放
    即a,d变化，s=1，其他为0；
    最终实现需要插值，下面再说。
"""
# resized = cv2.resize(image, (100, 400), interpolation=cv2.INTER_AREA)
# plt.imshow(resized, 'gray'), plt.show()


"""
2. 图像平移
    即a,d，s=1，s=1，p，q变化，其他为0；
"""
# M = np.float32([[1, 0, 25], [0, 1, 50]])  # 缺省了第三个维度，因为整个矩阵会随着第三个维度归一化，即第三个维度变成[0, 0, 1]^T
# shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# plt.imshow(shifted, 'gray'), plt.show()

"""
3. 图像镜像
    水平镜像：a=-1，d，s=1，p（原本x变为p-x），其他为0；
    垂直镜像：d=-1，a，s=1，q（原本y变为q-y），其他为0；
"""
# row = image.shape[0]
# col = image.shape[1]
# M1 = np.float32([[-1, 0, col], [0, 1, 0]])
# M2 = np.float32([[1, 0, 0], [0, -1, row]])
# shifted1 = cv2.warpAffine(image, M1, (image.shape[1], image.shape[0]))
# shifted2 = cv2.warpAffine(image, M2, (image.shape[1], image.shape[0]))
# plt.subplot(121), plt.imshow(shifted1, 'gray')
# plt.subplot(122), plt.imshow(shifted2, 'gray')
# plt.show()


"""
4. 图像旋转
    顺时针θ对应的： a = cosθ， b = sinθ， c = -sinθ， d = cosθ， s = 1，其他为0
"""
# (h, w) = image.shape[0:2]
# center = (h//2, w//2)   # 在python中10/3=3.3333  10//3=3
# M_rotate = cv2.getRotationMatrix2D(center, 45, 2.0)  # 逆时针45，第三个参数是放大倍数
# rotated = cv2.warpAffine(image, M_rotate, (w, h))
# plt.imshow(rotated, 'gray'), plt.show()


"""
5. 图像错切
    水平错切：ads=1，b变化
    垂直错切，ads=1，c变化
"""
# M = np.float32([[1, 0.1, 0], [0.1, 1, 0]])  # 缺省了第三个维度，因为整个矩阵会随着第三个维度归一化，即第三个维度变成[0, 0, 1]^T
# shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# plt.imshow(shifted, 'gray'), plt.show()


"""
6. 透视（近大远小）
    实现：原图像选4个点，然后把这四个点改变到希望的位置上，就有了对应的新的4个点
          然后根据这些点列线性方程组（之所以是线性的是因为他也是仿射变换来的），求解得到T矩阵参数。
          在这里除了s=1，其他都是未知量，共8个。
"""
pts1 = np.float32([[56, 65], [349, 52], [28, 335], [349, 349]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
plt.imshow(dst, 'gray'), plt.show()


"""
7. 插值
    前向插值：原图的1个像素按算法分配给后面好多个像素
    后向插值：后面的1个像素按算法来自于前面好多个像素
    最邻近插值:直接等于最近的pixel的灰度
    双线性插值：按照距离分配最近的4个pixel权重，然后相加
"""

"""
8. 几何校正
    1.若已知原图和期望的图像，标定点的方式标记4个点，然后用透视变换求取参数
    2.如果需要插值就插值
"""