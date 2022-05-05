import torch
from torch import nn


class Zj(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,input):
        output=input+1
        return output

zj1=Zj()  #使用Zj神经网络创建模板
x=torch.tensor(1.0)
output=zj1(x)   #把x输入到神经网络中，用output接收
print(output)