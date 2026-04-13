"""
案例：
    演示基础的图像操作

图像分类：
    二值图：        1通道，每个像素点由0，1组成
    灰度图：        1通道，每个像素点的范围：[0，255]
    索引图：        1通道，每个像素点范围：[0，255]，像素点表示颜色的索引
    RGB真彩图：     3通道，Red，Green，Blue，红绿蓝

"""
#导包：
import numpy as np
import matplotlib.pyplot as plt
import torch

#1.定义函数，绘制：全黑，全白图
def dm01():
    #1.定义全黑图片，像素点越接近0越黑，越接近255越白
    #HWC：H：高度，W：宽度，C：通道
    img1 = np.zeros((200,200,3))

    #2.绘制图片
    #plt.imshow(img1)
    #plt.show()

    #3.定义全白图像
    img2 = torch.full(size=(200,200,3),fill_value=255)
    plt.imshow(img2)
    plt.axis('off')   #关闭坐标轴
    plt.show()


#2.定义函数，加载图片

def dm02():
    #1.加载图片
    img1 = plt.imread('../shiJian/shuJuJi/nv.jpg')

    #2.保存图像
    plt.imsave('../shiJian/shuJuJi/img_copy.png',img1)

    #3.展示图像
    plt.imshow(img1)
    plt.show()

#3.测试
if __name__ == '__main__':
    #dm01()
    dm02()