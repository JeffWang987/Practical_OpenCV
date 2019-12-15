# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 11:24
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('coins.png', 0)
"""0.二值图像分析基础

        0. 目的：处理伪目标；处理目标物粘连，断开等。
        1. 手段：
                 0. 分割（区域分隔（标记法），边界分割（链码））；
                 1 .去噪，增强（数学形态学处理、面积滤波等）；
                 2. 特征提取（投影、交叉数与连接数、欧拉数、骨架、矩特征、几何特征（面积、周长、质心、圆形度、矩形度、长宽比等）、傅立叶描述子，矩描述等）
        2. 连通域：
                 0. 四连通：中间像素与上下左右像素至少有一个是相同的（黑或白）；
                 1. 八连通：中间像素与周围包裹的8个像素至少有一个是相同的（黑或白）；
        3. 连接域：连接在一起的黑色像素集合称为一个连通域（用来统计目标个数）
"""


"""1. 目标标记算法（数连接域个数）
    
        0. 步骤：
                0. 初始化：记当前像素标号Lab=0， 已标的总数N=0；
                1. 向右移动像素，直到遇到第一个黑色的像素，Lab=1；
                2. 继续移动像素，遇到第二个黑色像素，判断它的八连通域中是否存在已标记的：
                    0. 若存在标记，那么该像素Lab=已有标记的Lab；
                    1. 若不存在标记（可能只是暂时的，以下还会做修正），那么Lab=Lab+1， N=N+1；
                3. 继续移动像素，遇到黑色像素，判断它的八连通域中是否存在已标记的：
                    0.  若存在标记（可能有多个标记）：
                        0. 若所有标记相同，且那么该像素Lab=已有标记的Lab；
                        1. 若标记有不同的，那么取最小的Lab赋给所有的Lab，N=N-1，该像素自身也标为最小Lab；
                    1. 若不存在标记（可能只是暂时的，以下还会做修正），那么Lab=Lab+1， N=N+1；
                4. 继续移动像素直至完成。
"""
# 0. OpenCV中自带函数实现目标标记算法connectedComponent
img2 = np.array([
    [0, 255, 0, 0],
    [0, 0, 0, 255],
    [0, 0, 0, 255],
    [255, 0, 0, 0]], np.uint8)
N, labels = cv2.connectedComponents(img2)  # N为连接域个数， labels标记后的新图像
print(labels)

# 1. 将该算法应用于真实图像，需要把图像先二值化
(threshold, thresed) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
N2, labels2 = cv2.connectedComponents(thresed)

# 2. connectedComponentsWithStats
#    多了： centroid；：每个轮廓的的中心
#           states： 每个轮廓的[x0, y0, width, height, area]

img3 = np.array([
    [0, 255, 0, 0],
    [0, 0, 0, 255],
    [0, 0, 0, 255],
    [255, 0, 0, 0]], np.uint8)
N3, labels, stats, centroids = cv2.connectedComponentsWithStats(img3)








