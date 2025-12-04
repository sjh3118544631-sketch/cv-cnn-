"""
案例：
    演示自动微分模块，循环实现 计算梯度，更新参数

需求：
    求y = x**2 +20的极小值点 并打印y是最小值时 w的值（梯度）

解题步骤：
    1.    定义点 x = 10 requires_grad = True  dtype = torch.float32
    2.    定义函数： y =x**2+ 20
    3.    利用梯度下降法 循环迭代1000次 求最优解
    3.1   正向计算（前向传播）
    3.2   梯度清零 x.grad.zero_()
    3.3   反向传播
    3.4   梯度更新 x.data = x.data - 0.01 * x.grad

"""

#导包
import torch
from polars.testing.parametric.strategies.data import nulls

#1.    定义点 x = 10 requires_grad = True  dtype = torch.float32
w = torch.tensor(10,requires_grad=True,dtype=torch.float32)

#2.    定义函数： y =x**2+ 20
loss = w **2 +20
#3.    利用梯度下降法 循环迭代1000次 求最优解
print(f'开始权重初始值：{w},(0.01*w.grad):无，loss：{loss}')
#迭代100次找到最优解
for i in range(1,101):
   #3.1   正向计算（前向传播）
   loss = w **2 +20
   #3.2   梯度清零 x.grad.zero_()
   #第一次的时候还没有计算梯度，所以w.grad = None 要做非空判断
   if w.grad is  not None:
      w.grad.zero_()
   #3.3   反向传播
   loss.sum().backward()
   #3.4   梯度更新 x.data = x.data - 0.01 * x.grad
   w.data = w.data  - 0.01 * w.grad
   #3.5打印本次梯度更新后的权重参数结果
   print(f'第{i}次，权重初始值：{w},(0.01*w.grad):{0.01*w.grad:.5f},loss:{loss:.5f}')
#4.打印最终结果
print(f'最终结果 权重：{w},梯度：{w.grad:.5f},loss:{loss:.5f}')