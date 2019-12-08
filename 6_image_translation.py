# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 15:41
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm

import numpy as np
import cv2
import imutils   # 自己写的函数

image = cv2.imread("3_dinosaur.jpg")


"""
translate:图像位移
warpAffine: M * [x,y,1]^T   第三个参数是：图像输出尺寸
"""
# M1 = np.float32([[1, 0, 25], [0, 1, 50]])  # 缺省了第三个维度，因为整个矩阵会随着第三个维度归一化，即第三个维度变成[0, 0, 1]^T
# M2 = np.float32([[1, 0, -50], [0, 1, -90]])
# shifted1 = cv2.warpAffine(image, M1, (image.shape[1], image.shape[0]))
# shifted2 = cv2.warpAffine(image, M2, (image.shape[1], image.shape[0]))
# shifted3 = imutils.translate(image, 20, 100)
# cv2.imshow("shift down and right", shifted1)
# cv2.imshow("shift up and left", shifted2)
# cv2.imshow("function by myself", shifted3)
# cv2.waitKey(0)
"""
rotate:图像旋转
"""
# (h, w) = image.shape[0:2]
# center = (h//2, w//2)   # 在python中10/3=3.3333  10//3=3
# M_rotate1 = cv2.getRotationMatrix2D(center, 45, 2.0)  # 第三个参数是放大倍数
# M_rotate2 = cv2.getRotationMatrix2D(center, -90, 1.0)
# rotated1 = cv2.warpAffine(image, M_rotate1, (w, h))  # 逆时针
# rotated2 = cv2.warpAffine(image, M_rotate2, (w, h))
# rotated3 = imutils.rotate(image, 10)
# cv2.imshow("rotated1", rotated1)
# cv2.imshow("rotated2", rotated2)
# cv2.imshow("rotated3", rotated3)
# cv2.waitKey(0)

"""
resize:调整图像大小
"""
# r = 150.0 / image.shape[1]
# dim = (150, int(image.shape[0]*r))
# resized1 = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# resized2 = cv2.resize(image, (100, 400), interpolation=cv2.INTER_AREA)
# resized3 = imutils.resize(image, width=50)
# resized4 = imutils.resize(image, height=50)
# resized5 = imutils.resize(image, 20, 200)
# cv2.imshow("resized1", resized1)
# cv2.imshow("resized2", resized2)
# cv2.imshow("resized3", resized3)
# cv2.imshow("resized4", resized4)
# cv2.imshow("resized5", resized5)
# cv2.waitKey(0)

"""
flipping:反转图像
"""
# flipped1 = cv2.flip(image, 1)
# flipped2 = cv2.flip(image, 0)
# flipped3 = cv2.flip(image, -1)
# cv2.imshow("flipped1", flipped1)
# cv2.imshow("flipped2", flipped2)
# cv2.imshow("flipped3", flipped3)
# cv2.waitKey(0)

"""
crop:裁剪图像
"""
# cropped = image[30:120, 40:160]
# cv2.imshow("cropped", cropped)
# cv2.waitKey(0)

"""
image arithmetic:np和cv2对超过[0,255]范围的处理方式不一样，np是取溢出后的数，cv2是限幅操作。
"""
# M1 = np.ones(image.shape, dtype="uint8") * 100
# M2 = np.ones(image.shape, dtype="uint8") * 50
# added = cv2.add(image, M1)
# subtracted = cv2.subtract(image, M2)
# cv2.imshow("added", added)
# cv2.imshow("subtracted", subtracted)
# cv2.waitKey(0)

"""
bitwise operations:按位计算  AND OR XOR NOT
"""
# rectangle = np.zeros((300, 300), dtype="uint8")
# circle = np.zeros((300, 300), dtype="uint8")
# cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
# cv2.circle(circle, (150, 150), 150, 255, -1)
# cv2.imshow("rectangle", rectangle)
# cv2.imshow("circle", circle)
# bitwiseAnd = cv2.bitwise_and(rectangle, circle)
# bitwiseOr = cv2.bitwise_or(rectangle, circle)
# bitwiseXor = cv2.bitwise_xor(rectangle, circle)
# bitwiseNot = cv2.bitwise_not(rectangle)
# cv2.imshow("bitwiseAnd", bitwiseAnd)
# cv2.imshow("bitwiseOr", bitwiseOr)
# cv2.imshow("bitwiseXor", bitwiseXor)
# cv2.imshow("bitwiseNot", bitwiseNot)
# cv2.waitKey(0)


"""
masking:掩蔽运算
"""
# mask = np.zeros(image.shape[:2], dtype="uint8")
# (cX, cY) = (image.shape[1]//2, image.shape[0]//2)
# cv2.rectangle(mask, (cX-75, cY-75), (cX+75, cY+75), 255, -1)
# cv2.imshow("mask", mask)
# masked = cv2.bitwise_and(image, image, mask=mask)  # 先和自己And，然后再上mask
# cv2.imshow("masked", masked)
# cv2.waitKey(0)


"""
splitting and merging: 分离通道，合并通道
"""
# (B, G, R) = cv2.split(image)
# merged = cv2.merge([B, G, R])
# cv2.imshow("R", R)
# cv2.imshow("G", G)
# cv2.imshow("B", B)
# cv2.imshow("merged", merged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# zeros = np.zeros(image.shape[:2], dtype="uint8")
# merged_b = cv2.merge([B, zeros, zeros])
# cv2.imshow("merged_b", merged_b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
color space: 色域 RGB   HSV  L*a*b
"""
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# cv2.imshow("gray",gray)
# cv2.imshow("hsv", hsv)
# cv2.imshow("L*a*b", lab)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


