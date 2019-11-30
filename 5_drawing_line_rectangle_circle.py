# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 15:30
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np

canvas = np.zeros((300, 300, 3), dtype='uint8')
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)

cv2.line(canvas, (0, 0), (300, 300), green)
cv2.line(canvas, (300, 0), (0, 300), red, 5)


cv2.rectangle(canvas, (100, 100), (200, 200), green, -1)
cv2.rectangle(canvas, (200, 150), (300, 250), red, 3)

cv2.circle(canvas, (89, 89), 50, white, 1)

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

