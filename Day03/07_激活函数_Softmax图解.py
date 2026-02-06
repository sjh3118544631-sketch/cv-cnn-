"""
案例：
    绘制激活函数Softmax的 函数图像 和 导数图像

激活函数介绍
    激活函数的目的;
        给模型增加非线性功能，让模型（神经元）既可以做分类，也可以做回归问题
    激活函数的分类：
        Sigmoid
        ReLU
        Tanh
        Softmax

    Sigmoid激活函数：
        主要应用于二分类的输出层 ，且适用于 浅层神经网络（不超过5层）
        数据在[-6，6]之间有效果，在[-3，3]之间效果显著，会将数据映射到；[0,1]
        求导后范围在[0,0.25]

    Tanh:
        主要适用于 隐藏层，且适用于 浅层神经网络（不超过5层）
        数据在[-3,3]之间有效果，在[-1,1]之间效果明显，会将数据值映射到：[-1，1]
        求后范围在[0,1],相较于Sigmoid，收敛速度快
        
    ReLU:
        计算公式为：max（0，x），计算量相对以较小，训练成本低，多应用于隐藏层，且适合深度神经网络
        求导后，值要么是0要么是1，相较于Tanh，收敛速度更快
        默认情况下ReLU只考虑正样本，可以使用LeakyReLU,PReLU来考虑正负样本。

    Softmax：
        将多分类的结果以概率的形式展示，且概率和相加为1，最终选取概率值最大的分类作为最终结果
细节:
    绘制激活函数图像时出现以下提示需要将 anaconda3/Lib/site-packages/torch/lib目录下的libiomp5md.dll文件删除
    OMP： Error #15 ：Initializing libiomp5md.dll , but found libiomp5md.dll already initialized.
"""
#1.导包
import torch
import numpy as np
import matplotlib.pyplot as plt
from sympy.stats.rv import probability

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#1.定义张量，记录：分类数据
scores = torch.tensor([0.2,0.02,0.15,0.15,0.3,0.5,0.6,0.33,0.21,0.22])
#2. dim = 0 按行计算
probabilities = torch.softmax(scores,dim=0)
print(probabilities)