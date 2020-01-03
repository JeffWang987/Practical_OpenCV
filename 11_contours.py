# -*- coding: utf-8 -*-
# @Time    : 2019/11/29 23:04
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np
"""
本片文档学习了findcontours以及最小外接矩形以及普通矩形
"""

"""1.读取图像、转灰度、高斯滤波、Canny求边缘"""
image = cv2.imread("./picture/coins.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
edged = cv2.Canny(blurred, 30, 150)


"""2. 找轮廓、画轮廓"""
binary, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# param1: 最好输入copy而不是图像本身， param2：轮廓的形态， param3：RETR_EXTERNAL是最外面的边缘，RETR_LIST是全部轮廓， param4：近似轮廓的方法
# return: binary是二值化图像，contour是list表示轮廓本身， hierarchy代表轮廓属性，4个元素分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
print("I count {} coins in the image".format(len(contours)))
coins = image.copy()
cv2.drawContours(coins, contours, -1, (0, 255, 0), 2)  # 第三个参数是第几个轮廓， -1 代表全部， 然后颜色，线粗细
cv2.imshow("originla", image)
cv2.imshow("coins", coins)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""3. 普通裁剪部分轮廓

        0. 普通矩形框 (x, y, w, h) = cv2.boundingRect(c)
        1. 最小外包圆 ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
"""
# for (i, c) in enumerate(contours):  # i是第几个coin， c是contour的点
#     (x, y, w, h) = cv2.boundingRect(c)  # 通过轮廓的点数，获得一个box框住当前的轮廓，返回这个box的左下角坐标和宽度高度
#     print("Coin #{}".format(i+1))
#     coin = image[y:y+h, x:x+w]  # 坐标x,y的方向和 存储图像方向 相反
#     cv2.imshow("Coin", coin)
#
#     mask = np.zeros(image.shape[:2], dtype="uint8")
#     ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)  # 通过轮廓的点数，获得一个circle框住当前的轮廓，返回这个circle的圆心和半径
#     cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
#     mask = mask[y:y+h, x:x+w]
#     cv2.imshow("masked coin", cv2.bitwise_and(coin, coin, mask=mask))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


"""4. 最小外接矩形


        0. rect = cv2.minAreaRect(cnt)
        rect[0]为中心坐标
        rect[1][0]:width
        rect[1][q]:height
        rect[2]:angle(度数)，顺时针为正数，逆时针为负数
"""
rect = []
for (i, c) in enumerate(contours):  # i是第几个coin， c是contour的点
    rect = cv2.minAreaRect(c)
    print("angle:{}".format(rect[2]))
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
