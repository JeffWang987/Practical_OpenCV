# -*- coding: utf-8 -*-
# @Time    : 2019/12/19 10:52
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./picture/coins.png', 0)
angle = cv2.imread('./picture/chessboard.jpg')
beach = cv2.imread('./picture/beach.png')
coin = cv2.imread('./picture/coins.png')
"""本文档学习了图像分割，图像分割有以下几种方法。

        概述：将图像分割成连续的有意义的区域
        0. 基于阈值分割方法（实现简单，复杂度低；没有考虑空间信息，复杂背景效果不佳）
        1. 基于区域的分割方法（现在不常用）
        2. 基于边缘分割方法（都是求取像素梯度，利用多种算子模板）
        3. 基于角点分割方法（角点包含更多的信息，例如特征点匹配,且旋转、缩放不变。）
        4. 基于霍夫变化（频域）方法检测直线或圆
"""
"""0. 阈值方法
     0. 全局阈值（均值作为方差等）
     1. 自动阈值（迭代法、OTSU法、最小误差）
     2. 动态阈值（上述方法在光线不均匀时会失效，动态阈值又称自适应阈值，会根据局部区域的值来调整阈值）
"""
# 0. 全局阈值（均值作为方差等）
# T = np.round(img.mean())
# out = img.copy()
# out = 255 * (img > T)
# plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Origin')
# plt.subplot(122), plt.imshow(out, 'gray'), plt.title('Out'), plt.show()

# 1. 自动阈值（迭代法、OTSU法、最小误差）
#       0. 迭代法（随机初始化阈值、两类灰度求均值u1，u2，新的阈值为(u1+u2)/2）
# T = np.random.randint(0, 255, 1)
# for i in range(5):
#     u1 = np.sum(img[img >= T])/np.sum(img >= T)
#     u2 = np.sum(img[img < T])/np.sum(img < T)
#     T = (u1 + u2)/2
# out = 255 * (img > T)
# plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Origin')
# plt.subplot(122), plt.imshow(out, 'gray'), plt.title('Out'), plt.show()

#       1. OTSU类内方差最小，类间方差最大
# (threshold, out) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)  # cv2.THRESH_BINARY_INV是取反操作，也可以去除
# plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Origin')
# plt.subplot(122), plt.imshow(out, 'gray'), plt.title('Out'), plt.show()

# 2. 动态阈值
# out1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
# out2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
# # ADAPTIVE_THRESH_MEAN_C代表把一个点周围求均值作为该点的threshold，倒数2参数是奇数，代表11x11的size，倒数1参数是：调整threshold，减去该数
# # ADAPTIVE_THRESH_GAUSSIAN_C 使用带权的高斯模板卷积后作为threshold，其他一样
# plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Origin')
# plt.subplot(132), plt.imshow(out1, 'gray'), plt.title('Out1')
# plt.subplot(133), plt.imshow(out2, 'gray'), plt.title('Out2'), plt.show()


"""1. 区域增长方法（这个不常用，现在跳过，以后有缘遇到再说）

     0. 区域增长法
     1. 区域分割、合并（四叉树）
"""

"""2. 边缘方法

      0. 罗伯特（对角线）梯度fx=f(m,n)-f(m-1,n-1) y同理
      1.*拉普拉斯算子（四邻域），二阶差分算子
      2. 改进微分算子，中心差分fx = 1/2（f(x+1,y)-f(x-1,y)）；这样fx模板是[-1, 0, 1] fy模板是[-1, 0, 1]^T
      3. 普莱惠特(Prewitt)算子:先平滑后一阶差分，利用[1, 1, 1]和改进微分算子卷积结合。
      4,*Sobel算子:先平滑后一阶差分，利用[1, 2, 1]和改进微分算子卷积结合。
      5. Laplacian－Gaussian算子,二阶差分和高斯滤波结合。利用卷积性质，直接对高斯滤波二次差分就行了。
      6.*Canny算子（先高斯滤波，然后sobel，然后非极大值抑制，然后磁滞效应（双阈值）（https://opencv-python-tutorials.readthedocs.io/zh/latest/4.%20OpenCV%E4%B8%AD%E7%9A%84%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/4.7.%20Canny%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B/）
"""

# 1. 拉普拉斯算子（四邻域），二阶差分算子
# laplacian = cv2.Laplacian(img, cv2.CV_64F)
# plt.imshow(laplacian, 'gray'), plt.show()

# 4, Sobel算子:先平滑后一阶差分，利用[1, 2, 1]和改进微分算子卷积结合。
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# plt.subplot(121), plt.imshow(laplacian, 'gray')
# plt.subplot(122), plt.imshow(laplacian, 'gray'), plt.show()

# 6. Canny算子（先高斯滤波，然后sobel，然后非极大值抑制，然后磁滞效应（双阈值））（https://opencv-python-tutorials.readthedocs.io/zh/latest/4.%20OpenCV%E4%B8%AD%E7%9A%84%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/4.7.%20Canny%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B/）
# canny = cv2.Canny(img, 30, 150)
# plt.imshow(canny, 'gray'), plt.show()


"""3. 角点方法（可以用来匹配，匹配内容详情见https://opencv-python-tutorials.readthedocs.io/zh/latest/5.%20%E7%89%B9%E5%BE%81%E6%A3%80%E6%B5%8B%E5%92%8C%E6%8F%8F%E8%BF%B0/5.9.%20%E7%89%B9%E5%BE%81%E5%8C%B9%E9%85%8D/）

      0. Harris角点检测（旋转不变）
      1. SUSAN算法
      2. SIFT（旋转不变+缩放不变）注意，该算法现在已受专利保护，需要使用opencv3.4.3以下版本https://download.csdn.net/download/qq_35292250/11518052，或者c++有一个库叫opensift
      3. SURF（快速的sift）
      4. FAST(适用于应用的快速角点检测方法)
      5. BRIEF（他是一种特征描述符，用于快速匹配。而不是特征提取算法）
      6. 0RB(Oriented FAST and Rotated BRIEF)
"""
# 0. Harris角点检测(相当于求解某一个窗口的协方差矩阵的特征值，如果特征值在各个方向都大，说明有角点)
#                   (还有亚像素级的cv.cornerSubPix()，在这里就不用了)
# gray = np.float32(cv2.cvtColor(angle, cv2.COLOR_BGR2GRAY))  # 这个算法输入必须是浮点、灰度图像
# harris = cv2.cornerHarris(gray, 2, 3, 0.04)  # 角点检测的邻域大小; Sobel衍生物的孔径参数; Harris检测器自由参数
# center = harris > (0.01*harris.max())
# index = np.argwhere(harris > (0.01*harris.max()))  # 对于棋盘图片，其实只有49个角点，但是却检测出了441个，因为大多数角点处检测出了多个，可用形态学方法或非极大值抑制解决。
# for i in range(len(index)):
#     cv2.circle(angle, tuple(index[i]), 10, (0, 255, 0), -1)
# plt.imshow(angle), plt.show()

# 2. SIFT（先用图像金字塔，在同组塔之间做高斯差分，在不同尺度寻找局部极大值作为关键点。再为关键点指定方向参数。）
#       论文 Distinctive Image Featuresfrom Scale-Invariant Keypoints
# gray = cv2.cvtColor(beach, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()  # 注意，该算法现在已受专利保护，需要使用opencv3.4.3以下版本
# kp = sift.detect(gray, None)  # 在图像中找到关键点,返回的关键点是一个带有很多不同属性的特殊结构体，这些属性中包含它的坐标(x，y)，有意义的邻域大小，确定其方向的角度、指定关键点强度的响应等
# beach = cv2.drawKeypoints(gray, kp, beach)
# plt.imshow(beach), plt.show()

# 3. SURF （SURF比sift快，sift使用DoG对LoG进行近似，suft则更进一步，使用盒子滤波器（box_filter）对 LoG 进行近似）
#       论文 Surf: Speeded up robust features
# surf = cv2.xfeatures2d.SURF_create(50000)  # Hessian Threshold to 400,通过改变这个值可以减少或增多检测到的角点
# kp, des = surf.detectAndCompute(angle, None)
# angle = cv2.drawKeypoints(angle, kp, None, (255, 0, 0), 4)
# plt.imshow(angle), plt.show()

# 4. FAST（选取像素周围16个像素点，若有连续的n个像素高于当前点，则认为是角点；为了加速，先判断1，9，5，13和当前像素点的关系，至少有三个符合阈值要求，否则不是角点）
#           为了优化效果，还和机器学习（决策树）以及非极大值抑制结合。
# fast = cv2.FastFeatureDetector_create()
# kp = fast.detect(angle, None)
# img2 = cv2.drawKeypoints(angle, kp, None, color=(255, 0, 0))
# plt.imshow(img2), plt.show()
# print("Threshold: {}".format(fast.getThreshold()))
# print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
# print("neighborhood: {}".format(fast.getType()))
# print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

# 5. BRIEF(BRIEF 是一种特征描述符，它不提供查找特征的方法。所以使用特征检测器如SIFT)
#       简单来说 BRIEF 是一种对特征点描述符计算和匹配的快速方法。这种算法可以实现很高的识别率，除非出现平面内的大旋转
# SIFT = cv2.xfeatures2d.SIFT_create()
# brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# kp = SIFT.detect(angle, None)
# kp, des = brief.compute(angle, kp)

# 6. ORB（Oriented FAST and Rotated BRIEF）
# orb = cv2.ORB_create()
# kp = orb.detect(angle, None)
# kp, des = orb.compute(angle, kp)
# img2 = cv2.drawKeypoints(angle, kp, None, color=(0, 255, 0), flags=0)
# plt.imshow(img2), plt.show()


"""4. 霍夫变换

        0. 寻找直线原理：比如我有三点abc，过每一点都有无数种直线的可能。那么我在r-theta的极坐标空间（r=x*cos(theta)+y*sin(theta)）中，过每一点我以theta=1慢慢旋转这个直线，同时记录对应的r。然后对于三个点，寻找最匹配的r-theta
        1. 寻找圆的原理：(x-a)^2+(y-b)^2=r^2,有三个参数abr，所以对圆上多点，我们求每一点的abr，然后比较多个点的abr，那个匹配就让哪个作为参数
        可以看出hough变化有点暴力求解的意思，不过方法巧妙，角度清奇。为了减少运算，出现了概率Hough变换，cv2.HoughLinesP()
"""
# 0. 检测直线
# gray = cv2.cvtColor(angle, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # 必须要先求边缘
# lines = cv2.HoughLines(edges, 1, np.pi/180, 200)  # 输入图像；r的精度（像素），theta精度（弧度），最小投票数
# for line in lines:
#     rho, theta = line[0]  # lines返回的是rho和theta
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))  # 这里画图用1000个像素是因为：一旦我们检测出一段直线，就把这一段直线所在的全部直线一起画出来
#     y1 = int(y0 + 1000*a)
#     x2 = int(x0 - 1000*-b)
#     y2 = int(y0 - 1000*a)
#     cv2.line(angle, (x1, y1), (x2, y2), (0, 0, 255), 2)
# cv2.imshow('a', angle)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 1. 检测圆
# gray = cv2.cvtColor(coin, cv2.COLOR_BGR2GRAY)  # 输入灰度图像
# gray = cv2.medianBlur(gray, 5)
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,  # 灰度图像；梯度检测方法，输出图像和原图size比例，两个圆的最近距离
#                            param1=100, param2=30, minRadius=0, maxRadius=0)  # canny的高阈值参数，第二个参数越小，找到的圆越多
# circles = np.uint16(np.around(circles))  # 原本返回的圆是（a,b,r）float型
# for i in circles[0, :]:
#     cv2.circle(coin, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     cv2.circle(coin, (i[0], i[1]), 2, (0, 0, 255), 3)
# cv2.imshow('detected circles', coin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
