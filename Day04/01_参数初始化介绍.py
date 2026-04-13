"""
案例：
    演示参数初始化的7种方式
参数初始化的目的
    1.防止梯度消失 或者 梯度爆炸
    2.提升收敛速度
    3.打破对称性

参数初始化的方式：
    无法打破对称性的：
        全0，全1，固定值
    可以打破对称值：
        随机初始化，正态分布初始化，kaiming初始化，xavier初始化

总结:
    1.记忆 kaiming初始化，xavier初始化，全0初始化
    2.关于初始化的选择
        激活函数 ReLU及其系列：优先用kaiming
        激活函数非ReLU：优先用xavier
        如果是浅层网络：可以考虑随机初始化
"""

#1.导包
import torch.nn as nn   #nn: neural network

#1.均匀分布随机初始化
def dm01():
    #1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #2.对权重（w）进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(linear.weight)
    #3.对偏置（b）进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(linear.bias)
    #4.打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)

#2.固定值随机初始化
def dm02():
    #1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #2.对权重（w）进行随机初始化，设置固定值为3
    nn.init.constant_(linear.weight,3)
    #3.对偏置（b）进行随机初始化，设置固定值为5
    nn.init.constant_(linear.bias,5)
    #4.打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)

#3.全零初始化
def dm03():
    #1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #2.对权重（w）进行初始化，全零初始化
    nn.init.zeros_(linear.weight)
    #3.对偏置（b）进行初始化，全零初始化
    nn.init.zeros_(linear.bias)
    #4.打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)
#4.全一初始化
def dm04():
    #1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #2.对权重（w）进行初始化，全一初始化
    nn.init.ones_(linear.weight)
    #3.对偏置（b）进行初始化，全一初始化
    nn.init.ones_(linear.bias)
    #4.打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)
#5.正态分布随机初始化
def dm05():
    #1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #2.对权重（w）进行随机初始化，正态分布初始化（均值为0，标准差为1）
    nn.init.normal_(linear.weight)
    #3.对偏置（b）进行随机初始化，正态分布初始化（均值为0，标准差为1）
    nn.init.normal_(linear.bias)
    #4.打印生成结果
    print(linear.weight.data)
    print(linear.bias.data)
#6.kaiming初始化
def dm06():
    #6.1kaiming正态分布初始化
    #1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #2.对权重（w）进行随机初始化，正态分布初始化（均值为0，标准差为1）
    #nn.init.kaiming_normal_(linear.weight)
    #6.2kaiming均匀分布初始化
    #2.对权重（w）进行随机初始化，正态分布初始化（均值为0，标准差为1）
    nn.init.kaiming_uniform_(linear.weight)
    #3.打印结果
    print(linear.weight.data)
#7.xavier初始化
def dm07():
    # 7.1kaiming正态分布初始化
    # 1.创建一个线性层，输入维度5，输出维度3
    linear = nn.Linear(5, 3)
    # 2.对权重（w）进行随机初始化，正态分布初始化（均值为0，标准差为1）
    nn.init.xavier_normal_(linear.weight)
    # 7.2kaiming均匀分布初始化
    # 2.对权重（w）进行随机初始化，正态分布初始化（均值为0，标准差为1）
    nn.init.xavier_uniform_(linear.weight)
    # 3.打印结果
    print(linear.weight.data)



# 测试运行
if __name__ == '__main__':
    #dm01()  #均匀分布随机初始化
    #dm02()  #固定值随机初始化
    #dm03()  #全零初始化
    #dm04()   #全一初始化
    #dm05()   #正太分布初始化
    #dm06()   #kaiming初始化
    dm07()  #xavier初始化