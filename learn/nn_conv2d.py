import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./data",False,transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)

class Zj(nn.Module):
    def __init__(self):
        super(Zj, self).__init__()  #调用父类方法初始化
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
# self是必须写的,
#     定义其中一个卷积层，conv1
    def forward(self,x):
        x=self.conv1(x)
        # 调用conv1卷积层
        return x

zj=Zj()
# 实例化网络为zj
# print(zj)

writer=SummaryWriter("./p11")
step=0
for data in dataloader:
    imgs,targets=data
    output=zj(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input",imgs,step)
#  torch.Size([64, 6, 30, 30])
    output=torch.reshape(output,(-1,3,30,30))
    # 因为第一个参数bach确定，所以设置成-1
    writer.add_images("ouput",output,step)

    step=step+1

