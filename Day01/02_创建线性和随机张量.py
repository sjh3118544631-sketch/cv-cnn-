"""
案例：
    演示pytorch中如何创建线性和随即张量

涉及到的函数：
        torch.arange（） 和 torch.linspace() 创建线性张量
        torch.random.inital_seed() 和 torch.random.manual_seed()随机种子设置
        torch.rand/randn()创建随机浮点型张量
        torch.randint(low, high, size=()) 创建随机整数型张量

"""
import torch


#1.定义函数，演示创建线性张量
def dem01():
    #场景一：创建指定范围的张量；
    t1 = torch.arange(0,10,2)#起始值，结束值（包左不包右），步长
    print(f"t1:{t1},type{type(t1)}")
    print('-'*34)
    #场景二：创建指定范围的线性张量-> 等差数列
    t2 = torch.linspace(1,10,4)#从1开始10(包含)结束要4个数
    print(f"t2:{t2},type{type(t2)}")
    print('-'*34)

#2.定义函数，演示创建随机张量
def dem02():
    #step1:设置随机种子
    #torch.initial_seed()   #默认采用当前系统的时间戳作为随机种子
    torch.manual_seed(3)    #设置随机种子


    #step2:创建随机张量：
    #场景一：均匀分布的（0，1）随机张量
    t1 = torch.rand(size=(2,3))
    print(f"t1:{t1},type{type(t1)}")
    print('-' * 30)

    #场景二：符合正态分布的随机张量
    t2 = torch.randn(size=(2,3))
    print(f"t2:{t2},type{type(t2)}")
    print('-' * 30)

    # 场景三：创建整数随机张量
    t3 = torch.randint(low=1, high=10, size=(3, 5))
    print(f"t3:{t3},type{type(t3)}")
    print('-' * 30)




#3.测试函数
if __name__ == '__main__':
    #dem01()
    dem02()