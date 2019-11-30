# -*- coding: utf-8 -*-
# @Time    : 2019/11/28 16:17
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm

import numpy as np
import cv2
from matplotlib import pyplot as plt


def translate(image, x, y):
    """
    原本的warpaffine函数略复杂，因此集成化
    :param x:正代表向右
    :param y:正代表向下
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    """
    原本的warpaffine函数略复杂，因此集成化
    :param angle:正值逆时针
    :param center 按哪个点旋转
    :param scale:旋转完后放大倍数
    """
    (h, w) = image.shape[0:2]
    if center == None:
        center = (w//2, h//2)      #  注意center 是先w后h
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (h, w))
    return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    原本的warpaffine函数略复杂，因此集成化
    :param width:resize后的图像width
    :param height:resize后的图像height
    :param inter:插值方法
    """
    dim = (width, height)
    (h, w) = image.shape[:2]
    if width is None and height is None:
        print("You need to input size")
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def plot_histogram(image, title, mask=None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

