# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 15:00
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np

image = cv2.imread("3_dinosaur.jpg")
cv2.imshow("Original",image)
cv2.waitKey(0)

(b, g, r) = image[0][0]     # 颜色是tuple信息        # 取单个pixel [a][b]和[a, b]效果是一样的
print("Pixel at [0,0] - Red:{}, Green:{}, Blue:{}".format(r, g, b))
image[0][0] = (0, 0, 255)   # 改变颜色信息
(b, g, r) = image[0][0]     # 颜色是tuple信息
print("Now pixel at [0,0] - Red:{}, Green:{}, Blue:{}".format(r, g, b))
corner = image[0:100, 0:100]
cv2.imshow("Corner", corner)
image[0:100, 0:10] = (0, 0, 255)          # [0:a, 0:b] 和[0:a][0:b]是不一样的效果，原因暂时未知
cv2.imshow("Update", image)
cv2.waitKey(0)
