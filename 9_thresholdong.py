# -*- coding: utf-8 -*-
# @Time    : 2019/11/29 20:51
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np

image = cv2.imread("coins.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

"""
简单的二值化操作
"""
# (threshold, thresed) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
# # param2: threshold;  param3:利用param4方法超过阈值就设为这个； param4：二值化方法，如果大于就等于param3，下面那个方法如果小于xx
# # return threshold：threshold；  thresed：新的图像
# (threshold_inv, thresed_inv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("Threshold Binary", thresed_inv)
# cv2.imshow("Coins", cv2.bitwise_and(image, image, mask=thresed_inv))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
adaptive thresholding:自动阈值分割
均值和高斯阈值分割
"""
# thresed1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
# thresed2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
# # ADAPTIVE_THRESH_MEAN_C代表把一个点周围求均值作为该点的threshold，倒数2参数是奇数，代表11x11的size，倒数1参数是：调整threshold，减去该数
# # ADAPTIVE_THRESH_GAUSSIAN_C 使用带权均值作为threshold，其他一样
# cv2.imshow("Mean threshold", thresed1)
# cv2.imshow("Gaussian threshold", thresed2)
# cv2.imshow("original", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
阈值分割算法
OTSU：聚类的思想，把图像的灰度数按灰度级分成2个部分，使得两个部分之间的灰度值差异最大，每个部分之间的灰度差异最小
"""
# (threshold, thresed) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imshow("OTSU", thresed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



