# -*- coding: utf-8 -*-
# @Time    : 2019/11/29 20:28
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np

image = cv2.imread("3_dinosaur.jpg")

"""
averaging blur：均值滤波
"""
# blurred = np.stack([
#     cv2.blur(image, (3, 3)),
#     cv2.blur(image, (5, 5)),
#     cv2.blur(image, (7, 7))
# ])
# cv2.imshow("blurred", blurred[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
gaussian blur:高斯滤波
"""
# blurred = np.stack([
#     cv2.GaussianBlur(image, (3, 3), 0),  # standard deviation in X direction.
#     cv2.GaussianBlur(image, (5, 5), 0),
#     cv2.GaussianBlur(image, (7, 7), 0)
# ])
# cv2.imshow("blurred", blurred[2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
median blur:中值滤波
"""
# blurred = np.stack([
#     cv2.medianBlur(image, 3),
#     cv2.medianBlur(image, 5),
#     cv2.medianBlur(image, 7)
# ])
# cv2.imshow("blurred", blurred[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""
bilateral  这个原理不懂，可以减少噪声且保持edge
"""
# blurred = np.stack([
#     cv2.bilateralFilter(image, 5, 21, 21),
#     cv2.bilateralFilter(image, 7, 31, 31),
#     cv2.bilateralFilter(image, 9, 41, 41),
# ])
# cv2.imshow("blurred", blurred[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
