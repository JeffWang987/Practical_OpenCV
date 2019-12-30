# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 11:24
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./picture/coins.png', 0)
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
# img2 = np.array([
#     [0, 255, 0, 0],
#     [0, 0, 0, 255],
#     [0, 0, 0, 255],
#     [255, 0, 0, 0]], np.uint8)
# N, labels = cv2.connectedComponents(img2)  # N为连接域个数， labels标记后的新图像
# print(labels)
#
# # 1. 将该算法应用于真实图像，需要把图像先二值化
# (threshold, thresed) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# N2, labels2 = cv2.connectedComponents(thresed)
#
# # 2. connectedComponentsWithStats
# #    多了： centroid；：每个轮廓的的中心
# #           states： 每个轮廓的[x0, y0, width, height, area]
#
# img3 = np.array([
#     [0, 255, 0, 0],
#     [0, 0, 0, 255],
#     [0, 0, 0, 255],
#     [255, 0, 0, 0]], np.uint8)
# N3, labels, stats, centroids = cv2.connectedComponentsWithStats(img3)


"""2. 数学形态学 

        0. 基本思想：利用一定形态的结构元素去量度和提取图像中对应形状
        1. 二值形态学基本运算：膨胀、腐蚀、开闭运算
"""
# 0. 腐蚀Erosion操作：消除连通域的边界点，使边界向内收缩（前景缩小，背景扩大）
#    应用：消除小的背景噪声
#       0. 扫描原图，找到第一个像素为255的点
#       1. 将预先设置好的模板（模板的原点）移动到该点
#       2. 判断该模板所覆盖范围内的像素是否全为255
#           0. 如果全为255，则腐蚀后该像素保留255
#           1. 如果不全为255，则腐蚀后该像素为0
#       3. 重复1-2直至完成
# (threshold, thresed) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# kernel1 = np.ones((5, 5), np.uint8)
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构，和上面那个np创建的是一样的
# kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构, 因为coins是圆形，所以这里用这个比较好
# kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
# erosed1 = cv2.erode(thresed, kernel1)
# erosed2 = cv2.erode(thresed, kernel2)
# erosed3 = cv2.erode(thresed, kernel3)
# erosed4 = cv2.erode(thresed, kernel4)
# plt.subplot(151), plt.imshow(thresed)
# plt.subplot(152), plt.imshow(erosed1)
# plt.subplot(153), plt.imshow(erosed2)
# plt.subplot(154), plt.imshow(erosed3)
# plt.subplot(155), plt.imshow(erosed4), plt.show()

# 1. 膨胀dilation：使前景扩张，即目标物扩张
#    应用：使断开的目标物重新连接
#       0. 扫描原图，找到第一个像素为0的点
#       1. 将预先设置好的模板（模板的原点）移动到该点
#       2. 判断该模板所覆盖范围内的像素是否存在255
#           0. 如果存在255，则膨胀后该像素保留255
#           1. 如果不存在255，则膨胀后该像素为0
#       3. 重复1-2直至完成
# (threshold, thresed) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# kernel1 = np.ones((5, 5), np.uint8)
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构，和上面那个np创建的是一样的
# kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构, 因为coins是圆形，所以这里用这个比较好
# kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
# dilate1 = cv2.dilate(thresed, kernel1)
# dilate2 = cv2.dilate(thresed, kernel2)
# dilate3 = cv2.dilate(thresed, kernel3)
# dilate4 = cv2.dilate(thresed, kernel4)
# plt.subplot(151), plt.imshow(thresed)
# plt.subplot(152), plt.imshow(dilate1)
# plt.subplot(153), plt.imshow(dilate2)
# plt.subplot(154), plt.imshow(dilate3)
# plt.subplot(155), plt.imshow(dilate4), plt.show()

"""3. 开/闭运算

        0. 开运算：腐蚀+膨胀；用来分离物体，消除小区域
        1. 闭运算：膨胀+腐蚀：用来融合物体，消除小区域
        2. 通过下面的例子来理解（我们一共有9个coin）：
                0. 开运算较为保守，噪声多的coin不算了，只剩8个；
                1. 闭运算较为激进；
"""
# (threshold, thresed) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
# opening = cv2.morphologyEx(thresed, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(thresed, cv2.MORPH_CLOSE, kernel)
# plt.subplot(131), plt.imshow(thresed), plt.title('binary')
# plt.subplot(132), plt.imshow(opening), plt.title('opening')
# plt.subplot(133), plt.imshow(closing), plt.title('closing'), plt.show()


"""4. 形状特侦提取与描述

        0. 特征提取：投影（文字识别）、连通分量数与交叉点数、欧拉数（E=C-H，其中C为连通分量数,H为孔数；应用在数字识别，例如0-9中只有2的欧拉数=-1）
            0. 求质心，物理质心的离散化
            1. 细长物体朝向：如果物体是细长的，则可以把较长方向的轴定为物体的方向。通常，将最小二阶矩轴定义为较长物体的方向。
            2. 周长：法1：边界用隙码表示。求周长就是计算隙码的长度。
                     法2：周长用边界所占面积表示，也即边界点数之和。
            3. 轮廓提取：四邻域（八邻域）轮廓：如果领域全是物体颜色，那么变成背景色
            4. 面积：法1：边界内部（包括）前景像素个数之和
                     法2：用高数中的格林公式离散化
            5. 最小外接矩形：坐标系方向上的外接矩形
                             旋转物体使外接矩形最小
            6. 像素点距离：欧式、市区距离（绝对值，计算方便）、棋盘距离（max绝对值x，y）
            
        1. 图像描述：用一组描述子来表征图像中被描述物体的某些特征。
            0. 矩形度：反映物体对其外接矩形的充满程度，用面积比描述
            1. 最小外接矩形长宽比：区分细长物体
            2. 致密度（如果是圆的话该值=4pi）：度量圆形，周长平方与面积比
            3. 圆形性(不受区域平移、旋转、尺度变化影响，可推广到三维)：区域中心到边界点的平均距离/距离方差
            4. 形状度量（面积/平均距离平方），作用未知
            5. 链码：由方向数组成的数字串（没看懂，到时候百度）
            6. 傅里叶描述子，将坐标用复数表示，x轴作为实部，y轴作为虚部。然后傅里叶变换到频域。取前xx%的能量再返回到时域，可以还原图像，而且旋转不变。
            7. 矩特性（百度）零阶矩：面积；一阶：质心；二阶：主轴、辅轴；高阶：图像细节
            
            
            
"""













