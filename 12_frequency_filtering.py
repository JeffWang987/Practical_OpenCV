# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 11:29
# @Author  : Jeff Wang
# @Email   : jeffwang987@163.com   OR    wangxiaofeng2020@ia.ac.cn
# @Software: PyCharm

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("wave.png", 0)
"""
1. 前言！！！！！
    频域滤波：
        说到底最终可能是和空间域滤波实现相同的功能，比如实现图像的轮廓提取，
        在空间域滤波中我们使用一个拉普拉斯模板就可以提取，而在频域内，我们使用一个
        高通滤波模板（因为轮廓在频域内属于高频信号），可以实现轮廓的提取，后面也会
        把拉普拉斯模板(二阶梯度)频域化，会发现拉普拉斯其实在频域来讲就是一个高通滤波器。
    
    振幅：
        各个频率下的信号的决定程度有多大，如果某个频率的振幅越大，那么它对
        原始信号的的重要性越大，像上图，当然是w=1的时候振幅最大，说明它对总的信号
        影响最多（去掉w=1的信号，原始信号讲严重变形）。越往后面，也就是越高频，振幅
        逐渐减小，那么他们的作用就越小。
    
    相位：
        表示其实表面对应频率下的正弦分量偏离原点的程度.各个频率的分量相位都是0的话，
        那么每个正弦分量的最大值（在频率轴附近的那个最大值）都会落在频率轴为0上，
        然而上述图并不是这样。在说简单一点，比如原始信号上有个凹槽，正好是由某一频率的
        分量叠加出来的，那么如果这个频率的相位变大一点或者变小一点的话，带来的影响就会
        使得这个凹槽向左或者向右移动一下，也就是说，相位的作用就是精确定位到信号上一点的位置的。
"""
# illustrate1 = cv2.imread("Fourier1.png")
# cv2.imshow("傅里叶幅度分析", illustrate1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# illustrate2 = cv2.imread("Fourier2.png")
# cv2.imshow("傅里叶相位分析", illustrate2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
2. fft及ifft以及中心化操作
"""
# f = np.fft.fft2(img)  # 二维fft
# fshift = np.fft.fftshift(f)  # 将低频移动到图像中心
"""
    np.fft.fftshift(f)中心化操作：
        将其看成横纵两个方向的一维傅里叶变换，在每个方向上都会有高频信号和低频信号，那么傅里叶变换将低频信号放在了
        边缘，高频信号放在了中间，然而一副图像，很明显的低频信号多而明显，所以将低频信号采用一种方法移到中间，在时
        域上就是对f乘以（-1）^(M+N)，换到频域里面就是位置的移到了。  详情参考数字图像处理冈萨雷斯第三版148-149
"""
# s1 = np.log(np.abs(f))  # abs将复数变成实数，取对数的目的为了将数据变化到比较小的范围（i.e., 0~255）
# s2 = np.log(np.abs(fshift))
# f1shift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f1shift)
# img_back = np.abs(img_back)  # 在这里img_back理应又变成实数，但是由于数值计算的方法，这里还有十分小的虚数部分，所以要取abs
# plt.subplot(131), plt.imshow(s1, 'gray'), plt.title("original")
# plt.subplot(132), plt.imshow(s2, 'gray'), plt.title("center")
# plt.subplot(133), plt.imshow(img_back, 'gray'), plt.title('img_back')
# plt.show()


"""
3. 只用频域振幅或者相位来恢复时域图像
"""
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# f1shift = np.fft.ifftshift(np.abs(fshift))  # 取幅度
# f2shift = np.fft.ifftshift(np.angle(fshift))  # 取相位
# img_back1 = np.fft.ifft2(f1shift)
# img_back2 = np.fft.ifft2(f2shift)
# img_back1 = np.abs(img_back1)  # 把复数调整成实数
# img_back2 = np.abs(img_back2)
# img_back1 = (img_back1 - np.min(img_back1))/(np.max(img_back1) - np.min(img_back1))  # 调整大小以便显示
# img_back2 = (img_back2 - np.min(img_back2))/(np.max(img_back2) - np.min(img_back2))  # 注意很有趣的一件事，plt.imshow会自动根据数据的最大最小值然后安排到0-255，所以这句话放在这里没有，之所以放在这里是要保持这个意识。
# # 以下是将幅度和相位组合恢复图像
# s1 = np.abs(fshift)
# s1_angle = np.angle(fshift)
# s1_real = s1 * np.cos(s1_angle)
# s1_imag = s1 * np.sin(s1_angle)
# s2 = np.zeros(img.shape, dtype=complex)  # 现在需要虚数类型
# s2.real = s1_real
# s2.imag = s1_imag
# f2shift = np.fft.ifftshift(s2)
# img_back3 = np.fft.ifft2(f2shift)
# img_back3 = np.abs(img_back3)
# plt.subplot(221), plt.imshow(img, 'gray'), plt.title("original")
# plt.subplot(222), plt.imshow(img_back1, 'gray'), plt.title("only amplitude")
# plt.subplot(223), plt.imshow(img_back2, 'gray'), plt.title("only phase")
# plt.subplot(224), plt.imshow(img_back3, 'gray'), plt.title("phase+amplitude")
# plt.show()


"""
4. 结合A图像的幅度和B图像的相位。 
    这里我比较懒就不做了，就把3最后的复制一下就完了。
    结论：
        结合谁的相位，那么生成的图像就像谁。
    解释：
        可以理解振幅不过描述图像灰度的亮度，占用谁的振幅不过使得结果哪些部分偏亮或者暗而已，
        而图像是个什么样子是由它的相位决定的。相位描述的是一个方向，方向正确了，那么最终的结果离你的目的就不远了。
"""


"""
5. 频域滤波器：图像在变换加移动中心后，从中间到外面，频率上依次是从低频到高频的
    5.1 高通滤波器，可以提取轮廓
    5.2 低通滤波器，模糊图像
    5.3 带通滤波器，tradeoff
"""
# centerX = img.shape[1]//2
# centerY = img.shape[0]//2
"""高通"""
# mask = np.ones(img.shape, dtype="uint8")
# mask[centerY-30:centerY+30, centerX-30:centerX+30] = 0
"""低通"""
# mask = np.zeros(img.shape, dtype="uint8")
# mask[centerY-20:centerY+20, centerX-20:centerX+20] = 1
"""带通"""
# mask1 = np.ones(img.shape, dtype="uint8")
# mask2 = np.zeros(img.shape, dtype="uint8")
# mask1[centerY-8:centerY+8, centerX-8:centerX+8] = 0
# mask2[centerY-20:centerY+20, centerX-20:centerX+20] = 1
# mask = mask1 * mask2
# f1 = np.fft.fft2(img)
# fshift = np.fft.fftshift(f1)
# fshift = fshift * mask
# f2shift = np.fft.ifftshift(fshift)
# img_new = np.fft.ifft2(f2shift)
# img_new = np.abs(img_new)
# img_new = (img_new-np.min(img_new))/(np.max(img_new)-np.min(img_new))  # 道理同上
# plt.subplot(121), plt.imshow(img, "gray"), plt.title("original")
# plt.subplot(122), plt.imshow(img_new, "gray"), plt.title("high pass")
# plt.show()


"""
6. 可以把时域上的一些卷积模板放到频域上看一看。
"""
# laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 拉普拉斯是求边缘的算子
# f = np.fft.fft2(laplacian)
# fshift = np.fft.fftshift(f)
# img_fre = np.abs(fshift)
# plt.imshow(img_fre, 'gray')
# plt.show()  # 高频率波，所以中心是0，周围是1
