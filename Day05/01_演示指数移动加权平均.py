"""
案例：
    演示近三十天，天气分布情况

结论：
    针对于β（调节权重系数）来讲，其值越大说明：月以来指数加权平均，越不依赖本地的梯度值，数据就越：平缓


"""

import torch
import matplotlib.pyplot as plt
from scipy.special.cython_special import betaln
from sympy.codegen.ast import continue_
from sympy.physics.units import temperature
ELEMENT_NUMBER = 30

#1.实际平均温度
def dm01():
    #固定随机种子
    torch.manual_seed(0)
    #产生30天的随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER, ])*10
    print(temperature)
    #绘制平均温度
    days = torch.arange(1,ELEMENT_NUMBER + 1,1)
    plt.plot(days,temperature,color = 'r')
    plt.scatter(days,temperature)
    plt.show()

#2.指数加权平均温度
def dm02(beta = 0.9):
    #固定随机种子
    torch.manual_seed(0)
    #产生30天随机温度
    temperature = torch.randn(size=[ELEMENT_NUMBER,])*10
    print(temperature)

    exp_weight_avg = []
    #idx从1开始
    for idx , temp in enumerate(temperature,1):
        # 第一个元素的 EWA 值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        #第二个元素的EWA 值等于上一个EWA 乘以β +当前气温乘以（1-β）
        #idx-2:2-2=0,exp_weight_avg列表中第一个值的下标值
        new_temp = exp_weight_avg[idx - 2] * beta + (1-beta) * temp
        exp_weight_avg.append(new_temp)

    days = torch.arange(1,ELEMENT_NUMBER+1,1)
    plt.plot(days,exp_weight_avg,color = 'r')
    plt.scatter(days,temperature)
    plt.show()

if __name__ == '__main__':
    dm01()
    dm02(0.5)
    dm02(0.9)