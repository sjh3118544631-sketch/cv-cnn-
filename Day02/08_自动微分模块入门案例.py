"""
案例：
    演示自动微分模块如何求导
回顾：
    权重更新公式：
        W新 = W旧 - 学习率 * 梯度

        梯度 = 损失函数求导

    关于损失函数求导我们不需要手动计算，以为非常常用，所以pytorch模块内置有自动微分模块
    ，专门实现针对于不同的损失函数求导，从而实现结合反向传播，
    更新 权重参数W 和 偏执参数 b
细节：
    只有标量张量才能求导，且大多数底层操作都是 浮点型，记得转型。

"""
#导包
import torch

#1.定义变量，记录：初始的权重W（旧）
#参1：初始值，参2：是否自动微分（求导），参3：数据类型
w = torch.tensor(10,requires_grad=True,dtype=torch.float)

#定义loss变量，表示损失函数。
loss = 2 * w**2   #loss -> 2w² -> 求导：4w

#3.打印梯度函数类型（了解）
#print(f'梯度函数类型：{type(loss._grad_fn)}')   #< class 'MulBackward0'>
#print(loss.sum())

#4.计算梯度，梯度 = 损失函数的导数，计算完毕后，会记录到w.grad属性中
loss.sum().backward()  #保证loss是1个标量
#loss.backward()        #这里因为本身就是标量，可以不写sum（）

#5. 带入 权重更新公式：W新 = W旧 - 学习率 * 梯度
w.data = w.data - 0.01 * w.grad

#6.打印最终结果
print(f'更新后的权重：{w}')   #9.6