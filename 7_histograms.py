# -*- coding: utf-8 -*-
# @Time    : 2019/11/29 9:58
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
from matplotlib import pyplot as plt
import numpy as np
import cv2
image = cv2.imread("beach.png")
import imutils

"""
画灰度图像的histogram
"""
# hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # image变成list, channels[0, 1, 2]代表所有通道， mask， histSize， ranges of pixel values
# plt.figure()
# plt.title("Gray Hist")
# plt.xlabel("Bins")
# plt.ylabel("# of pixels")
# plt.plot(hist)
# plt.xlim([0,256])
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
画彩色图像三个通道的histogram
"""
# chans = cv2.split(image)
# colors = ("b", "g", "r")
# plt.figure()
# plt.title("color histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of pixels")
# for (chan, color) in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])
# plt.show()


"""
histogram equalization:直方图均衡化
"""
# (b, g, r) = cv2.split(image)   # 历程是把图像转为灰度再进行操作，我分成三个通道分别操作再融合
# eq_b = cv2.equalizeHist(b)
# eq_g = cv2.equalizeHist(g)
# eq_r = cv2.equalizeHist(r)
# eq = cv2.merge([eq_b, eq_g, eq_r])
# cv2.imshow("Histogram equation", eq)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
histogram with mask
"""
# mask = np.zeros(image.shape[:2], dtype="uint8")
# cv2.rectangle(mask, (15, 15), (130, 100), 255, -1)
# cv2.imshow("mask", mask)
# masked = cv2.bitwise_and(image, image, mask=mask)
# cv2.imshow("masked", masked)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# imutils.plot_histogram(image, "masked histogram", mask)
