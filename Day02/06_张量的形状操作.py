"""
案例：
    演示张量的形状操作
涉及到的API：
    reshape()     在不改变张量内容的前提下，对其形状做改变（有桌面补充）
    unsqueeze()   在指定的轴上增加一个（1）维度，等价于：生维。
    squeeze()     删除所有为1的维度，等价于：降维。
    transpose()
    permute()
    view()
    contiguous()
    is_contiguous()

需要掌握的函数：
    reshape(),unsqueeze(),permute(),view()
"""

#导包
import torch

#指定随机种子
torch.manual_seed(24)
#1. 定义函数，演示：reshape()
def dm01():
    #1.定义1个两行三列的张量
    t1 = torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape},row:{t1.shape[0]},columns:{t1.shape[1]},{t1.shape[-1]}')
    print('-'*30)
    #2.通过reshape（）函数，把t1->3行2列，1行6列，6行1列
    #t2 = t1.reshape(3,2)
    #t2 = t1.reshape(1,6)
    t2 = t1.reshape(6,1)
    print(f't2:{t2},shape:{t2.shape},row:{t2.shape[0]},columns:{t2.shape[1]},{t2.shape[-1]}')
    print('-' * 30)
   #3.尝试通过reshape（）函数，把t1-> 2行5列的结果
    #t2 = t1.reshape(2,5)    #报错，转之前攻击2*3=6个元素，转之后共计2*5 = 10个元素，不一致
    #print(f't2:{t2}')

#2. 定义函数，演示：unsqueeze()，squeeze()
def dm02():
    #1.定义2行3列的张量
    t1 = torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape}')    #(2,3)
    print('-' * 30)
    #2.在0维上，添加一个维度
    t2 = t1.unsqueeze(0)
    print(f't2:{t2},shape:{t2.shape}')   # (2,3)->(1,2,3)
    print('-' * 30)
    # 3.在1维上，添加一个维度
    t3 = t1.unsqueeze(1)
    print(f't3:{t3},shape:{t3.shape}')  # (2,3)->(2,1,3)
    print('-' * 30)
    # 4.在2维上，添加一个维度
    t4 = t1.unsqueeze(2)
    print(f't4:{t4},shape:{t4.shape}')  # (2,3)->(2,3,1)
    print('-' * 30)
    # 5.在3维上，添加一个维度     会报错
    #t5 = t1.unsqueeze(3)
    #print(f't5:{t5},shape:{t5.shape}')  # (2,3)->(2,3,*,,1)
#3. 定义函数，演示：transpose()，permute()


#4. 定义函数，演示：view()，contiguous()，is_contiguous()


#5.测试
if __name__ == '__main__':
    #dm01()
     dm02()