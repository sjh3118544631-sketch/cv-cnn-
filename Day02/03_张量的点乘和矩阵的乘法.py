"""
案例：
    演示张量的点乘 和 矩阵的乘法操作
点乘：
    要求：两个张量的维度保持一致，对应元素直接做出相应的操作
    API:
        t1*t2
        t1.mul(t2)      multiply: 乘法

矩阵乘法：
    要求：两个张量，第一个张量的列数，等于第二个·张量的列数（A列 = B行）
    结果：A行B列
    API:
        t1 @ t2
        t1.matmul(t2)
"""
#导包
import torch

#1.定义函数，演示张量：点乘（行列数要一致）
def dm01():
    #1.定义张量 两行三列
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't1:{t1}')
    #2.定义张量 两行三列
    t2 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't2:{t2}')
    #演示点乘
    #t3 = t1.mul(t2)
    t3 = t1*t2
    print(f't3:{t3}')

#2.定义函数，演示张量：矩阵乘法（A列 = B行，结果：A行B列）
def dm02():
    # 1.定义张量 两行三列
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1:{t1}')
    # 2.定义张量 三行三列
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6],[7,8,9]])
    print(f't2:{t2}')
    # 演示点乘
    # t3 = t1.matmul(t2)
    t3 = t1 @ t2
    print(f't3:{t3}')


#3.测试
if __name__ == '__main__':
   # dm01()
    dm02()