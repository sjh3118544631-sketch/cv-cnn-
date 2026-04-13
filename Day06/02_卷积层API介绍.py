"""
案例：
    演示卷积层API，用于提取图像的 局部特征 ，获取 ：特征图（Feature Map）

卷积神经网络介绍：
    概述：
        全称叫：Convolutional neural network ,即：包含卷积层的神经网络
    组成：
        卷积层（ Convolutional）：
            用于提取图像的 局部特征 ，结合 卷积核（每个卷积核 = 1个神经元）实现，处理后的结果叫特征图
        池化层（Pooling）:
            用于 降维 ，降采样
        全连接层（Full Connected,fc ，linear）：
            用于 预测结果， 并输出结果的

    特征图计算方式：
        N = (W - F + 2*P)/S + 1
        N:输出图像的大小（特征图大小）
        W:输入图像的大小
        F:卷积核的大小
        P:填充的大小
        S:步长

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#1.定义函数， 用于完成图像的加载，卷积，特征图可视化操作

def dm01():
    #1.加载RGB真彩图
    img = plt.imread('../shiJian/shuJuJi/nv.jpg')
    #2.打印读取到的图像信息
    #print(f'img:{img},shape:{img.shape}')

    #3.把图像形状从 HWC -> CHW 思路：img -> 张量 -> 转换维度
    img2 = torch.tensor(img,dtype = torch.float)
    img2 = img2.permute(2,0,1)

    #4.因为这里只有1张图，所以我们给他在增加一个维度，从 CHW-> (1,C,H,W),1张3通道的416*245像素图
    img3 = img2.unsqueeze(dim = 0)

    #5.创建卷积层对象，提取 特征图
    #参1:输入图像的通道数 参2: 输出图像的通道数（几个特征图）参3: 卷积核的大小参4: 步长参5: 填充
    conv = nn.Conv2d(3,4,3,2,0)

    #6.具体的卷积计算
    conv_img = conv(img3)

    #7.打印卷积后的结果 (1张4通道的207*122像素的图像)
    print(f'conv_img:{conv_img},shape:{conv_img.shape}')

    #8.查看提取到的 4个 特征图
    img4 = conv_img[0]

    #9.把上述的图从 CHW -> HWC
    img5 = img4.permute(1,2,0)

    #10.可视化第一个通道的特征图
    feature1 = img5[:,:,0].detach().numpy()         #第0通道（即：第一通道的）像素图
    plt.imshow(feature1)
    plt.show()






#3.测试
if __name__ == '__main__':
    dm01()