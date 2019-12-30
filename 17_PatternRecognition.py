# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 11:44
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('./picture/wave.png')

"""本文档学习了图像识别基础，主要包含：
        
        0. 模式和模式类（基础知识）
        1. 模板匹配
        2. Bayes分类器
        3. 应用举例
"""

"""0. 模式和模式类（模式类(Class)：一类事物的代表；模式(Pattern)：某一事物的具体表现）
    
        0. 定义：模式类(Class)：一类事物的代表；模式(Pattern)：某一事物的具体表现；
                 如：数字0,1,2,3,4,5,6,7,8,9是模式类，用户任意手写的一个数字则是模式，是数字的具体化。
        1. 假设有N个样本，每个样本n个特征，针对这些特诊，有一些描述：均值，方差，协方差，协方差矩阵、
"""

"""1. 模板匹配

        0. 

"""