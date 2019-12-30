# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 15:47
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm


import cv2

'''如果需要在cmd中运行的话
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
'''


image = cv2.imread('./picture/dinosaur.jpg')
print("height: {} pixels".format(image.shape[0]))   # 注意，由于矩阵的存储形式，先行后列，故shape[0]是高
print("width: {} pixels".format(image.shape[1]))
print("channels: {}".format(image.shape[2]))

cv2.imshow("dinosaur.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./picture/dinosaur.png", image)
