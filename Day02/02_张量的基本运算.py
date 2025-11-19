"""
案例：
    演示张量的基本运算
涉及到的API：
    add(),sub(),mul(),div(),neg()   ->加减乘除，取反
    add_(),sub_(),mul_(),div_(),neg_()   ->功能同上，只不过可以修改源数据类似于Pandas中的inplace = Ture

需要记忆的：
    1.可以用+，-，*，/符号来代替上述的加减乘除函数
    2.如果是张量和数值运算，则：该数值会和张量中的每个值依次进行对应的运算

"""
#1.导包
import torch
#2.创建张量
t1 = torch.tensor([1,2,3])
#3.演示
t2 = t1.add(10)   #不改变源数据  -> t2 = t1+10
#t2 = t1.add_(10)  #改变源数据  -> t1 = t1+10  或t1+ = 10
print(f't1:{t1},type:{type(t1)}')
print(f't2:{t2},type:{type(t2)}')
#其他的同上
t3 = t1 - 1  #减
print(f't3:{t3}')
t4 = t1 * 2  #乘
print(f't4:{t4}')
t5 = t1 / 2  #除   有小数 -> [0.5,1,1.5]
print(f't5:{t5}')
t6 = -t1#t6 = t1.neg()  取反
print(f't6:{t6}')
t7 = t1 // 2 #整除
print(f't7:{t7}')
