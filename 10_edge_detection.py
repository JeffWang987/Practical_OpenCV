# -*- coding: utf-8 -*-
# @Time    : 2019/11/29 22:11
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np

image = cv2.imread("./picture/coins.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
"""
LAPLASIAN :一阶微分边缘检测
"""
# lap = cv2.Laplacian(image, cv2.CV_64F)  # 不是uint8 而是64的float，因为梯度有正负梯度，防止负梯度没用
# lap = np.uint8(np.absolute(lap))
# cv2.imshow("Laplasian", lap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
SOBEL :边缘检测
"""
# sobelX = np.uint8(np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0)))
# sobelY = np.uint8(np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1)))
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)
# cv2.imshow("sobel_X", sobelX)
# cv2.imshow("sobel_Y", sobelY)
# cv2.imshow("sobel", sobelCombined)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
Canny :边缘检测（边缘噪声少）:
    步骤:
        首先对图像选择一定的Gauss滤波器进行平滑滤波;
        利用已有的一阶偏导算子计算梯度。一般用sobel;
        非极大值抑制;
        双阈值法抑制假边缘，连接真边缘(即参数的最后一条) 
    参数:
        低于阈值1的像素点会被认为不是边缘;
        高于阈值2的像素点会被认为是边缘；
        在阈值1和阈值2之间的像素点,若与第2步得到的边缘像素点相邻，则被认为是边缘，否则被认为不是边缘。
"""
# image = cv2.GaussianBlur(image, (5, 5), 0)
# canny = cv2.Canny(image, 30, 150)
# cv2.imshow("canny", canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
