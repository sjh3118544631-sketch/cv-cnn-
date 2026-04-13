"""
案例：
    演示神经网络搭建流程

深度学习案例的4个步骤
    1.准备数据
    2.搭建神经网络
    3.模型训练
    4.模型测试

神经网络搭建流程
    1.定义一个类，继承：nn.Module
    2.在__init__（）方法中，搭建神经网络
    3.在forward（）方法中完成前向传播

"""

#导包
import torch
import torch.nn as nn
from torchsummary import summary

#todo: 1.搭建神经网络，即：自定义继承nn.Module
class ModelDemo(nn.Module):
    #todo：1. 在init魔法方法中，完成初始化父类成员 及 神经网络的搭建

    def __init__(self):
        #1.1 初始化父类成员
        super().__init__()
        #1.2 搭建神经网络 ——> 隐藏层 + 输出层
        #隐藏层1： 输入特征3，输出特征3
        self.linear1 = nn.Linear(3,3)
        #隐藏层2： 输入特征3，输出特征数2
        self.linear2 = nn.Linear(3,2)
        #输出层： 输入特征2，输出特征2
        self.output = nn.Linear(2,2)

        #1.3对隐藏层进行参数初始化
        #隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        #隐藏层2
        nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    #todo:1.2 向前传播： 输入层 -> 隐藏层 -> 输出层

    def forward(self,x):
        #1.1 第一层 隐藏层计算： 加权求和 + 激活函数（sigmoid）
        #分解版写法
        #x = self.linear1（x）
        #x = torch.sigmoid(x)
        #合并版写法
        x = torch.sigmoid(self.linear1(x))

        #1.2 第二层隐藏层：加权求和 + 激活函数（relu）
        x = torch.relu(self.linear2(x))
        #1.3输出层： 加权求和 + 激活函数（softmax）
        #dim = -1 表述按行计算（一行样本一行样本的处理）， dim = 0 表示按列计算
        x = torch.softmax(self.output(x),dim= -1)

        #1.4：返回预测值
        return x

#todo: 2.模型训练

def train():
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #1. 创建模型对象
    my_model = ModelDemo().to(device)
    #print(f'my_model:{my_model}')

    #2.创建数据集样本随机生成
    data = torch.randn(size=(5,3)).to(device)
    print(f'data:{data}')
    print(f'data.shape:{data.shape}')                   #5行3列
    print(f'data.requires_grad:{data.requires_grad}')   #False 判断能否自动微分，默认为False

    #3.调用神经网络模型 -> 进行模型训练
    output = (my_model(data))    #底层自动调用了 forward（）方法，进行前向传播
    print(f'output:{output}')
    print(f'output.shape:{output.shape}')  # 5行2列
    print(f'output.requires_grad:{output.requires_grad}')
    print('-' * 30)

    #4.计算 和 查看模型参数
    print('========================计算模型参数=========================')
    #参1：（神经网络）模型对象，参2：输入数据的维度
    summary(my_model,input_size=(5,3),device=device.type)

    print('========================查看模型参数=========================')
    for name,param in my_model.named_parameters():
        print(f'name:{name}')
        print(f'device: {param.device}')
        print(f'param:{param}\n')



#todo:3.测试
if __name__ == '__main__':
    train()