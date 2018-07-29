# coding:utf8
from torch import nn
from .BasicModule import BasicModule, Flat


# 首先定义VGG块，VGG最主要的思想就是多加卷积，且卷积后图的大小不变，图的大小变化靠池化来完成
# input：num_convs：卷积的次数
#       in_channels：输入的通道数
#       out_channels：输出的通道数
#
# output：返回Sequential的VGG模块
def vgg_block(num_convs, in_channels, out_channels):
    '''
    VGG子模块
    '''
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)] # 定义第一层

    for i in range(num_convs-1): # 定义后面的很多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
        
    net.append(nn.MaxPool2d(2, 2)) # 定义池化层
    return nn.Sequential(*net)

# 该函数主要用来堆叠VGG块，VGG卷积网络就是由多个VGG块和最后的全连接层实现
# input：num_convs：VGG块中卷积的次数（由数组来表示）
#       channels：通道数（由[输入通道，输出通道]这样的数组所组成的大数组表示
#       out_channels：输出的通道数
#
# output：返回VGG卷积部分的网络
def vgg_stack(num_convs, channels):
    '''
    堆叠VGG块
    '''
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


class vggNet(BasicModule):
    
    def __init__(self, num_classes):
        super(vggNet, self).__init__()
        
        self.model_name = 'vggNet'
        
        self.features = vgg_stack([1, 1, 2, 2, 2], [[3, 64], [64, 128], [128, 256], [256, 512], [512, 512]])
        
        self.flatten = Flat()

        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x



