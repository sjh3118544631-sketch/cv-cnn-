"""
案例：
    代码演示 随机失活

正则化的作用：
    缓解模型过拟合情况

正则化的方式：
    L1正则化：权重可以变为0，相当于降维
    L2正则化:权重可以无限接近于0
    DropOut：随机失活，每批次样本训练时，随即让一部分神经元死亡，防止一些特征对结果的影响较大（防止过拟合）
    BN（批量归一化）

"""
#导包
import torch
import torch.nn as nn
from torch.nn.functional import dropout


#1.定义函数，演示：随机失活（DropOut）
def dm_01():
    #1.创建隐藏层输出结果
    t1 = torch.randint(0,10,size=(1,4)).float()

    #2.进行下一层 加权求和 和 激活函数 计算
    #2.1 创建全连接层（充当线性层）
    #参1：输入维度，参2：输出维度
    linear1 = nn.Linear(4,5)

    #2.2加权求和
    l1 = linear1(t1)
    print(f'l1:{l1}')

    #2.3激活函数
    output = torch.relu(l1)
    print(f'output:{output}')

    #3.对激活函数进行随机失活dropout处理 -> 只有训练阶段有，测试阶段没有
    dropout = nn.Dropout(p = 0.4) #每个神经元都有40%的概率死亡
    #具体的 随机失活动作
    d1 = dropout(output)
    print(f'd1:{d1}')


#2.测试：
if __name__ == '__main__':
    dm_01()
